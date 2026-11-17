from settings import ROOT,ASSETS,CODE,SEED,save,log,digest
from runtime import seed_all,require_gpu,load_backbone,data,clear,state_digest
from backbone import install_lora,adapter_state,load_adapter,merge_lora,forward,prediction_rows
from risk import SpatiotemporalRisk
from sampling import build_sampler,generate,features,ar_alternative,Collector
from repeats import spans,metrics,requested,KINDS
import gc,json,random,re,subprocess,sys,time,torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
import flash_experiment as flash

def load_ar():
    ar=AutoModelForCausalLM.from_pretrained(flash.QWEN,local_files_only=True,torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
    return ar,AutoTokenizer.from_pretrained(flash.QWEN,local_files_only=True)

def gsm_prompt(record,tok):
    s=tok.decode(record['ids'][:record['prefix']],skip_special_tokens=False)
    m=re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>',s,re.S)
    assert m,'GSM user prompt extraction failed'
    return dict(id=record['id'],prompt=m.group(1),kind='gsm8k',reference_answer=tok.decode(record['ids'][record['prefix']:],skip_special_tokens=True))

def general(topic,zh,en,index,group):
    lang='zh' if index%2==0 else 'en';story=(index//2)%2==1
    if lang=='zh':prompt=(f'写一个关于{zh}的小故事，有明确结尾。' if story else f'介绍{zh}的主要组成、工作原理和使用方法。')
    else:prompt=(f'Write a short story about {en}, with a clear ending.' if story else f'Describe the main parts, working principle and uses of {en}.')
    return dict(id=f'{group}-{topic}-{index}',prompt=prompt,kind='general',topic=topic,language=lang)

def prepare(tok):
    d=data();bank=json.loads((ASSETS/'evaluation-prompts.json').read_text(encoding='utf-8'))
    general_test=[r for r in bank if r['kind']=='general'][:16]
    test=general_test+[r for r in bank if r['kind']=='gsm8k'][:12]+[r for r in bank if r['kind']=='intent'][:2]+[r for r in bank if r['kind']=='intent'][4:6]
    train_topics=[('kettle','电热水壶','an electric kettle'),('stapler','订书机','a stapler'),('projector','投影仪','a projector'),
        ('compass','指南针','a compass'),('typewriter','打字机','a typewriter'),('zipper','拉链','a zipper'),
        ('waterwheel','水车','a waterwheel'),('thermos','保温瓶','a thermos'),('loom','织布机','a loom'),
        ('metronome','节拍器','a metronome'),('windmill','风车','a windmill'),('telescope','天文望远镜','a telescope')]
    dev_topics=[('elevator','电梯','an elevator'),('microscope','显微镜','a microscope'),('sewing','缝纫机','a sewing machine'),('printer','打印机','a printer')]
    trains=[]
    for block in range(3):
        prompts=[gsm_prompt(r,tok) for r in d['train'][block*8:block*8+8]]
        prompts += [general(*train_topics[(block*4+j//2)%12],j+block*8,'train') for j in range(8)]
        trains.append(prompts)
    dev=[gsm_prompt(r,tok) for r in d['dev'][:8]]+[general(*dev_topics[j//2],j,'dev') for j in range(8)]
    assert len(test)==32 and len(dev)==16
    assert not ({r['id'] for r in dev}|{r['id'] for b in trains for r in b}) & {r['id'] for r in test}
    save('prompts.json',dict(train=trains,dev=dev,test=test))
    files={p.name:digest(p) for p in CODE.glob('*.py')}
    save('protocol-fixed.json',dict(seed=42,updates=96,refresh_updates=[0,32,64],free_rollouts=48,
        output_slots=192,source_sha256=files,train_original_diffusion_examples=192,dev=16,test=32,
        risk_types=KINDS,auxiliary='.1 balanced BCE + .1 detached risk weighted softplus(bad-alt+.5)',
        runtime_excludes_flagged_boundary_token=True,training_free_rollouts_controller_disabled=True,
        dense_feature_replay_on_native_cached_inputs=True,pilot_not_blind=True))
    return d,trains,dev,test

def samples_from(row,c,tok):
    if len(c.frames)<4:return []
    assert len(c.frames)==len(c.frontiers)
    events=spans(row['tokens'],tok,c.protect)
    positive=[]
    for s in events:
        t=next((i for i,n in enumerate(c.frontiers) if n>=s['end']),None)
        if t is not None and t>=3:positive.append(t)
    indices=[]
    if positive:indices.append(positive[0])
    rng=random.Random(42+len(row['tokens']))
    indices.append(rng.randrange(3,len(c.frames)))
    result=[]
    for t in dict.fromkeys(indices):
        latest=c.frames[t][0,c.prefix:]
        horizon=c.frontiers[min(t+3,len(c.frontiers)-1)]
        y=torch.zeros(192,3);valid=latest.eq(tok.mask_token_id)&(torch.arange(192)<horizon)
        forecast=[]
        for s in events:
            if s['end']<=horizon and valid[s['start']:s['end']].any():
                y[s['start']:s['end'],s['kind']]=1;forecast.append(s)
        repair=sorted(forecast,key=lambda s:(-s['kind'],s['start']))[:1]
        result.append(dict(frames=[f.clone() for f in c.frames[t-3:t+1]],prefix=c.prefix,
            y=y,valid=valid,repair=repair,tokens=row['tokens'],step=t,horizon=horizon,
            actual_native_inputs=True,no_gold_response_prefix=True))
    return result

def risk_loss(logits,y,valid):
    losses=[]
    for k in range(3):
        raw=F.binary_cross_entropy_with_logits(logits[:,k],y[:,k],reduction='none')
        pos=valid&y[:,k].bool();neg=valid&~y[:,k].bool()
        pieces=[raw[m].mean() for m in (pos,neg) if m.any()]
        if pieces:losses.append(torch.stack(pieces).mean())
    return torch.stack(losses).mean() if losses else logits.sum()*0

def replay(model,risk,projection,sample,tok,training):
    frames=[f.to('cuda') for f in sample['frames']];prefix=sample['prefix']
    fs=[features(model,f,projection,prefix,grad=training) for f in frames]
    if training:
        for f in fs:f.retain_grad()
    h=torch.stack([f[0] for f in fs])[None]
    logits=risk(h,frames[-1][:,prefix:].eq(tok.mask_token_id))[0]
    y=sample['y'].to('cuda');valid=sample['valid'].to('cuda')
    detect=risk_loss(logits,y,valid);margin=logits.sum()*0
    repair=sample.get('repair',[])
    if training and repair and sample.get('alternative') is not None:
        s=repair[0];p=s['start'];bad=sample['tokens'][p];alt=sample['alternative']
        canvas=frames[-1].clone()
        # The generated prefix is the actual free-running prefix; there is no gold alignment.
        canvas[0,prefix:prefix+p]=torch.tensor(sample['tokens'][:p],device='cuda')
        canvas[:,prefix+p:]=tok.mask_token_id
        hidden,_=forward(model,canvas,checkpointed=True)
        output=model.lm_head(prediction_rows(hidden)[0,prefix+p]).float()
        gate=.25+.75*logits[p:s['end'],s['kind']].sigmoid().max().detach()
        margin=gate*F.softplus(output[bad]-output[alt]+.5)
    loss=.1*detect+.1*margin
    return loss,dict(risk=float(detect.detach()),margin=float(margin.detach()),
        positives=[int((y[:,k].bool()&valid).sum()) for k in range(3)],valid=int(valid.sum())),fs,(logits.detach().cpu(),y.cpu(),valid.cpu())

def train():
    seed_all();require_gpu();model,tok,_=load_backbone();install_lora(model)
    risk=SpatiotemporalRisk().to('cuda');projection=torch.load(ASSETS/'projection.pt',weights_only=True).to('cuda')
    ar,ar_tok=load_ar();d,trains,_,_=prepare(tok);patched=build_sampler()
    # Exact stock/disabled/collector parity on a full 192-slot native generation.
    sentinel=trains[0][8]
    model.eval();risk.eval()
    a,_=generate(model,ar,tok,ar_tok,sentinel,projection)
    b,_=generate(model,ar,tok,ar_tok,sentinel,projection,patched=patched)
    c,observer=generate(model,ar,tok,ar_tok,sentinel,projection,patched=patched,collect=True)
    assert a['raw_ids']==b['raw_ids']==c['raw_ids']
    save('stock-parity.json',dict(all192_equal=True,collector_equal=True,actual_frames=len(observer.frames)))
    params=[p for p in model.parameters() if p.requires_grad]+list(risk.parameters())
    optimizer=torch.optim.AdamW(params,lr=2e-5,weight_decay=.01)
    base_versions={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    initial=state_digest(adapter_state(model));updates=[];proof=False
    for block,prompts in enumerate(trains):
        ar.to('cuda');model.eval();risk.eval();pool=[]
        log('collecting',block=block,policy_update=block*32,total=16)
        cached=ROOT/'trajectory-samples'/'block-0.pt'
        reuse_initial=(block==0 and cached.exists())
        if reuse_initial:
            pool=torch.load(cached,weights_only=True,map_location='cpu')
            assert all(s['policy_update']==0 for s in pool)
            assert {s['record_id'] for s in pool}<={r['id'] for r in prompts}
            log('reused_initial_original_policy_rollouts',samples=len(pool),positive_samples=sum(bool(s['repair']) for s in pool))
        for i,r in enumerate([] if reuse_initial else prompts):
            row,c=generate(model,ar,tok,ar_tok,r,projection,patched=patched,collect=True)
            row.update(record=r,policy_update=block*32,metrics=metrics(row['tokens'],tok,requested(r['prompt'])))
            save(f'rollouts/{block:02d}-{i:02d}.json',row)
            ss=samples_from(row,c,tok)
            for s in ss:
                s['record_id']=r['id'];s['policy_update']=block*32
                if s['repair']:
                    p=s['repair'][0]['start'];full=s['frames'][-1][0,:s['prefix']].to('cuda')
                    pref=torch.cat((full,torch.tensor(s['tokens'][:p],device='cuda',dtype=torch.long)))
                    alt,detail=ar_alternative(ar,tok,ar_tok,pref,s['tokens'][p])
                    s['alternative']=alt;s['ar_supervision']=detail
                pool.append(s)
            log('collecting',block=block,policy_update=block*32,completed=i+1,total=16,samples=len(pool),
                positive_samples=sum(bool(s['repair']) for s in pool))
            clear(model)
        assert pool,'No four-state free trajectories'
        (ROOT/'trajectory-samples').mkdir(exist_ok=True)
        torch.save(pool,ROOT/'trajectory-samples'/f'block-{block}.pt')
        ar.to('cpu');clear(model);model.train();risk.train()
        positive=[s for s in pool if s['repair']];negative=[s for s in pool if not s['repair']]
        for local in range(32):
            step=block*32+local;optimizer.zero_grad(set_to_none=True);diff=0.
            for micro in range(2):
                record=d['train'][step*2+micro];rng=random.Random(SEED+step*2+micro)
                gold=torch.tensor([record['ids']],device='cuda');prob=rng.uniform(.1,.9)
                selected=torch.tensor([i>=record['prefix'] and rng.random()<prob for i in range(gold.size(1))],device='cuda')
                if not selected.any():selected[record['prefix']]=True
                noisy=gold.clone();noisy[0,selected]=tok.mask_token_id
                hidden,_=forward(model,noisy,checkpointed=True)
                out=model.lm_head(prediction_rows(hidden)[0,selected])
                loss=F.cross_entropy(out.float(),gold[0,selected],reduction='sum')/(prob*(gold.size(1)-record['prefix'])*2)
                loss.backward();diff+=float(loss.detach());del hidden,out,loss
            eligible=positive if positive and local%4!=3 else (negative or pool)
            sample=eligible[local%len(eligible)]
            loss,meta,fs,_=replay(model,risk,projection,sample,tok,True)
            assert torch.isfinite(loss)
            if not proof:
                grads=torch.autograd.grad(loss,[p for p in model.parameters() if p.requires_grad],retain_graph=True,allow_unused=True)
                norm=sum(float(g.float().square().sum()) for g in grads if g is not None)**.5
                assert norm>0;del grads
            loss.backward()
            if not proof:
                feature_norms=[float(f.grad.norm()) for f in fs];assert all(v>0 for v in feature_norms)
                save('gradient-proof.json',dict(auxiliary_lora_gradient_norm=norm,four_frame_gradient_norms=feature_norms,
                    four_distinct_inputs=len({f.numpy().tobytes() for f in sample['frames']})==4,
                    risk_has_gradient=any(p.grad is not None for p in risk.parameters())))
                assert len({f.numpy().tobytes() for f in sample['frames']})==4;proof=True
            torch.nn.utils.clip_grad_norm_(params,1.)
            for g in optimizer.param_groups:g['lr']=2e-5*min(1.,(step+1)/8)
            optimizer.step();updates.append(dict(update=step+1,diffusion=diff,auxiliary=float(loss.detach()),**meta))
            del loss,fs
            if local%4==0 or local==31:log('training',update=step+1,total=96,**meta)
        torch.save(dict(adapter=adapter_state(model),risk=risk.state_dict(),updates=(block+1)*32),ROOT/'checkpoint.pt')
        save(f'block-{block}-complete.json',dict(policy_update_source=block*32,pool_size=len(pool),positive_samples=len(positive),
            adapter_sha256=state_digest(adapter_state(model))))
    assert base_versions=={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
    save('training-complete.json',dict(updates=updates,initial_adapter=initial,final_adapter=state_digest(adapter_state(model)),
        base_frozen=True,free_rollouts=48,refresh_updates=[0,32,64],proof=proof))
    log('training_complete',updates=96)

def load_treatment():
    model,tok,_=load_backbone();install_lora(model)
    ck=torch.load(ROOT/'checkpoint.pt',weights_only=True,map_location='cpu');assert ck['updates']==96
    load_adapter(model,ck['adapter']);merge_lora(model)
    risk=SpatiotemporalRisk().to('cuda').eval();risk.load_state_dict(ck['risk']);risk.requires_grad_(False)
    model.eval().requires_grad_(False)
    return model,tok,risk

@torch.inference_mode()
def calibrate():
    seed_all();require_gpu();model,tok,risk=load_treatment();ar,ar_tok=load_ar()
    projection=torch.load(ASSETS/'projection.pt',weights_only=True).to('cuda');patched=build_sampler()
    dev=json.loads((ROOT/'prompts.json').read_text(encoding='utf-8'))['dev'];samples=[]
    for i,r in enumerate(dev):
        row,c=generate(model,ar,tok,ar_tok,r,projection,patched=patched,collect=True)
        save(f'dev/{r["id"]}.json',dict(record=r,**row))
        samples.extend(samples_from(row,c,tok));clear(model);log('calibrating',completed=i+1,total=16)
    ar.to('cpu');clear(model);scores=[]
    for s in samples:
        _,_,_,(z,y,v)=replay(model,risk,projection,s,tok,False)
        scores.append(dict(p=z.sigmoid(),y=y,valid=v))
    assert scores
    thresholds=[];details=[]
    for k in range(3):
        p=torch.cat([x['p'][:,k][x['valid']] for x in scores]);y=torch.cat([x['y'][:,k][x['valid']] for x in scores]).bool()
        candidates=[]
        for t in (.3,.4,.5,.6,.7,.8,.9):
            pred=p>=t;tp=int((pred&y).sum());fp=int((pred&~y).sum());fn=int((~pred&y).sum())
            candidates.append(dict(threshold=t,tp=tp,fp=fp,fn=fn,f1=2*tp/max(1,2*tp+fp+fn)))
        best=max(candidates,key=lambda a:(a['f1'],-a['fp'],a['threshold'])) if y.any() else dict(threshold=.5,no_dev_positives=True)
        thresholds.append(best['threshold']);details.append(dict(kind=KINDS[k],positive=int(y.sum()),negative=int((~y).sum()),chosen=best,grid=candidates))
    torch.save(scores,ROOT/'dev-risk-scores.pt');save('thresholds.json',dict(thresholds=thresholds,details=details,dev_only=True))
    log('calibration_complete',thresholds=thresholds)

def number(text):
    from decimal import Decimal
    if '####' in text:text=text.rsplit('####',1)[-1]
    else:
        boxed=re.findall(r'\\boxed\{([^{}]+)\}',text)
        if boxed:text=boxed[-1]
    values=re.findall(r'-?\d[\d,]*(?:\.\d+)?',text)
    return str(Decimal(values[-1].replace(',','')).normalize()) if values else None

@torch.inference_mode()
def evaluate(arm):
    seed_all();require_gpu()
    if arm=='complete':model,tok,risk=load_treatment()
    else:model,tok,_=load_backbone();risk=None
    model.eval().requires_grad_(False);ar,ar_tok=load_ar()
    projection=torch.load(ASSETS/'projection.pt',weights_only=True).to('cuda');patched=build_sampler()
    th=json.loads((ROOT/'thresholds.json').read_text(encoding='utf-8'))['thresholds']
    records=json.loads((ROOT/'prompts.json').read_text(encoding='utf-8'))['test']
    for i,r in enumerate(records):
        path=ROOT/'eval'/arm/(r['id']+'.json')
        if path.exists():continue
        row,_=generate(model,ar,tok,ar_tok,r,projection,patched=patched if risk else None,risk=risk,thresholds=th)
        row.update(record=r,arm=arm,metrics=metrics(row['tokens'],tok,requested(r['prompt'])))
        if r['kind']=='gsm8k':row['correct']=number(row['text'])==number(r['reference_answer'])
        if r['kind']=='intent':row['intent_correct']=row['text'].strip()==r['expected_text']
        save(f'eval/{arm}/{r["id"]}.json',row);clear(model)
        log('evaluating',arm=arm,completed=i+1,total=32,actions=len(row['actions']))
    save(f'eval-{arm}-complete.json',dict(outputs=32,fresh=True))

if __name__=='__main__':
    {'train':train,'calibrate':calibrate,'original':lambda:evaluate('original'),'complete':lambda:evaluate('complete')}[sys.argv[1]]()

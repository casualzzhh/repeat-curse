from settings import ROOT,save
import importlib,inspect,hashlib,signal,time,torch
from backbone import forward,prediction_rows
from repeats import spans,requested
import flash_experiment as flash

MODULE=importlib.import_module('guided_diffusion.guided_diff_utils')
SAMPLER=MODULE.assisted_block_diffusion_generate
SOURCE=inspect.getsource(SAMPLER)

def build_sampler():
    source=SOURCE
    needle='    while (seq == mask_id).any():\n';assert source.count(needle)==1
    source=source.replace(needle,needle+'        if _controller is not None: _controller.observe(seq,diffusion_step)\n')
    needle='\n        if not stopped:\n';assert source.count(needle)==1
    source=source.replace(needle,'''
        if _controller is not None:
            _rewound = _controller.after(seq,last_non_mask,diffusion_step,verifier,verification_stats)
            if _rewound is not None:
                last_non_mask = _rewound
                stopped = False
'''+needle)
    ns=dict(MODULE.__dict__);ns['_controller']=None
    exec(compile(source,str(ROOT/'runtime_sampler.py'),'exec'),ns)
    (ROOT/'runtime_sampler.py').write_text(source)
    save('sampler-provenance.json',dict(stock_sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),
        patched_sha256=hashlib.sha256(source.encode()).hexdigest(),stock_not_mutated=True,
        native_first_mismatch_commits_dream=True,repair_explicitly_commits_ar=True))
    return ns['assisted_block_diffusion_generate'],ns

def content(seq,prefix,mask,eos):
    values=seq[0,prefix:].tolist();out=[]
    for v in values:
        if v in (mask,eos):break
        out.append(v)
    return out

def features(model,seq,projection,prefix,grad=False):
    saved=[(l.self_attn.block_cache,getattr(l.self_attn,'max_length',None)) for l in model.model.layers]
    init=model.model.cache_initialized
    try:
        _,f=forward(model,seq,projection,prefix,checkpointed=grad,middle_only=True)
        return f
    finally:
        for layer,(cache,length) in zip(model.model.layers,saved):
            layer.self_attn.block_cache=cache
            if length is not None:layer.self_attn.max_length=length
        model.model.cache_initialized=init

@torch.no_grad()
def ar_alternative(ar,tok,ar_tok,prefix_ids,old):
    mapper=MODULE.TokenMapper(tok,ar_tok)
    q=mapper.dream_to_qwen_tensor(prefix_ids,ar.device)
    out=ar(q[None],use_cache=False)
    logp=out.logits[0,-1].float().log_softmax(-1)
    vals,idx=logp.topk(64);special=set(tok.all_special_ids)
    for rank,(v,qid) in enumerate(zip(vals.tolist(),idx.tolist())):
        token=int(mapper.qwen_to_dream_id(qid))
        if token==old:continue
        if token in special and not (token==tok.eos_token_id and rank==0):continue
        return token,dict(ar_rank=rank+1,ar_log_probability=v,old_token=old,ar_fresh_prefix=True)
    return None,dict(no_safe_ar_alternative=True)

class Collector:
    def __init__(self,model,ar,tok,ar_tok,projection,prefix,prompt,risk=None,thresholds=None):
        self.model=model;self.ar=ar;self.tok=tok;self.ar_tok=ar_tok;self.projection=projection
        self.prefix=prefix;self.protect=requested(prompt);self.risk=risk;self.thresholds=thresholds
        self.frames=[];self.frontiers=[];self.actions=[];self.scores=[];self.attempted=set();self.risk_observations=[]
    def observe(self,seq,step):
        self.frames.append(seq.detach().cpu().clone());self.step=step
    def after(self,seq,last,step,verifier,stats):
        tokens=content(seq,self.prefix,self.tok.mask_token_id,self.tok.eos_token_id)
        self.frontiers.append(len(tokens))
        if self.risk is None or self.protect or len(self.frames)<4:return None
        ev=spans(tokens,self.tok)
        ev=[s for s in ev if s['end']>max(0,(self.frontiers[-2] if len(self.frontiers)>1 else 0)) and (s['kind'],s['start']) not in self.attempted]
        if not ev:return None
        history=torch.stack([features(self.model,f.to('cuda'),self.projection,self.prefix)[0] for f in self.frames[-4:]])[None]
        risk=self.risk(history,seq[:,self.prefix:].eq(self.tok.mask_token_id))[0].sigmoid()
        del history
        for s in sorted(ev,key=lambda s:(-s['kind'],s['start'])):
            self.attempted.add((s['kind'],s['start']))
            score=float(risk[s['start']:s['end'],s['kind']].max())
            self.scores.append(dict(**s,score=score,step=step))
            if len(self.actions)>=8 or score<self.thresholds[s['kind']]:continue
            target=self.prefix+s['start'];old=int(seq[0,target]);original_prompt=seq[:,:self.prefix].clone()
            # Materialized later copy is genuinely masked; dependent suffix is invalidated.
            repair=seq.clone();repair[:,target:]=self.tok.mask_token_id
            hidden,_=forward(self.model,repair)
            fresh_logits=self.model.lm_head(prediction_rows(hidden)[0,target]).float()
            fresh_draft=int(fresh_logits.argmax());del hidden,fresh_logits
            chosen,detail=ar_alternative(self.ar,self.tok,self.ar_tok,repair[0,:target],old)
            if chosen is None:continue
            repair[0,target]=chosen
            seq.copy_(repair);assert torch.equal(seq[:,:self.prefix],original_prompt)
            length=seq.size(1);self.model._experiment_length=length
            flash.reset(self.model,length,True)
            out=flash.call(self.model,seq,0,True,freeze_at=max(self.prefix,target-64));del out
            counts={k:stats[k] for k in ('ar_model_calls','rejection_correction_calls') if k in stats}
            verifier.reset();stats.update(counts)
            self.actions.append(dict(**s,score=score,step=step,target=target,old=old,new=chosen,
                fresh_dream_draft=fresh_draft,changed=old!=chosen,remasked_suffix=True,
                ar_choice_committed=True,dream_cache_rebuilt=True,ar_cache_reset=True,**detail))
            # Drop temporal history spanning a branch edit; obtain four new real inputs.
            self.frames=[]
            return target
        return None

@torch.inference_mode()
def generate(model,ar,tok,ar_tok,record,projection,patched=None,risk=None,thresholds=None,collect=False):
    ids=tok.apply_chat_template([{'role':'user','content':record['prompt']}],add_generation_prompt=True,return_tensors='pt').to('cuda')
    prefix=ids.size(1);length=prefix+192
    model.model.cache_initialized=False
    for l in model.model.layers:l.self_attn.block_cache=None
    model._experiment_length=length;model._mask_id=tok.mask_token_id
    config=flash.AssistedDiffusionConfig(max_length=length,max_new_tokens=192,mask_token_id=tok.mask_token_id,
        eos_token_id=tok.eos_token_id,temperature=0.,sampling_strategy='deterministic',verify_batch_size=32,
        use_sliding_window_caching=True,sliding_window_size=(64,192),return_dict_in_generate=True,output_history=False)
    controller=Collector(model,ar,tok,ar_tok,projection,prefix,record['prompt'],risk,thresholds) if collect or risk is not None else None
    sampler=SAMPLER
    if patched:sampler,ns=patched;ns['_controller']=controller
    previous=signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(RuntimeError('watchdog; not a successful short answer')))
    signal.alarm(360);start=time.monotonic()
    try:out=sampler(dream_model=model,ar_model=ar,input_ids=ids,attention_mask=torch.ones_like(ids),config=config,dream_tokenizer=tok,ar_tokenizer=ar_tok)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        if patched:ns['_controller']=None
    assert torch.equal(out.sequences[:,:prefix],ids)
    tokens=content(out.sequences,prefix,tok.mask_token_id,tok.eos_token_id)
    row=dict(tokens=tokens,text=tok.decode(tokens,skip_special_tokens=True),raw_ids=out.sequences[0,prefix:].cpu().tolist(),
        seconds=time.monotonic()-start,prefix=prefix,prompt_unchanged=True,
        actions=controller.actions if controller else [],scores=controller.scores if controller else [])
    return row,controller

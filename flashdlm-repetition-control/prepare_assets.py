"""Validate bundled public split and regenerate the exact fixed projection."""
import argparse,hashlib,json,math,random
from pathlib import Path

PROJECTION_SHA='800c33dc4f6576331c1cd74c26dc22c2e7d66298d25f8aeeac2a322179685df5'

def build_split(train,test,tok,ntrain=1024,ndev=64,ntest=64):
    train=list(train);test=list(test);rng=random.Random(42);rng.shuffle(train);rng.shuffle(test)
    prepared=[];skipped=0
    for record in train:
        prefix=tok.apply_chat_template([{'role':'user','content':record['question']}],add_generation_prompt=True,tokenize=True)
        if len(prefix)>96:skipped+=1;continue
        answer=tok.encode(record['answer'],add_special_tokens=False)+[tok.eos_token_id]
        prepared.append(dict(id=hashlib.sha256(record['question'].encode()).hexdigest(),ids=(prefix+answer)[:192],
            prefix=len(prefix),truncated=len(prefix)+len(answer)>192))
        if len(prepared)==ntrain+ndev:break
    assert len(prepared)==ntrain+ndev,'Insufficient eligible training examples'
    train_ids={hashlib.sha256(r['question'].encode()).hexdigest() for r in train}
    selected=[]
    for record in test:
        ident=hashlib.sha256(record['question'].encode()).hexdigest();assert ident not in train_ids
        selected.append(dict(id=ident,**record))
        if len(selected)==ntest:break
    return dict(train=prepared[:ntrain],dev=prepared[ntrain:],test=selected,
        skipped_long_train_questions=skipped,train_dev_ids_disjoint=True,all_train_test_questions_disjoint=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw-train',type=Path);parser.add_argument('--raw-test',type=Path)
    args=parser.parse_args()
    from settings import ASSETS,option
    ASSETS.mkdir(parents=True,exist_ok=True)
    original=json.loads((ASSETS/'gsm8k_split.json').read_text(encoding='utf-8'))
    assert [len(original[k]) for k in ('train','dev','test')]==[1024,64,64]
    if args.raw_train or args.raw_test:
        if not(args.raw_train and args.raw_test):parser.error('Supply both raw files')
        from transformers import AutoTokenizer
        hashes={'train.jsonl':hashlib.sha256(args.raw_train.read_bytes()).hexdigest(),
                'gsm8k_test.jsonl':hashlib.sha256(args.raw_test.read_bytes()).hexdigest()}
        assert hashes==original['source_hashes'],'Raw source order/bytes differ from the reference experiment'
        tok=AutoTokenizer.from_pretrained(str(option('dream_model','Dream-org/Dream-v0-Instruct-7B')),
            revision=str(option('dream_revision','05334cb9faaf763692dcf9d8737c642be2b2a6ae')),trust_remote_code=True,local_files_only=True)
        read=lambda p:[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
        rebuilt=build_split(read(args.raw_train),read(args.raw_test),tok)
        for name in ('train','dev','test'):assert rebuilt[name]==original[name],f'{name} reconstruction mismatch'
        print('Raw GSM8K reconstruction matches the bundled split')
    import torch
    p=torch.randn(3584,128,generator=torch.Generator().manual_seed(501))/math.sqrt(3584)
    sha=hashlib.sha256(p.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert sha==PROJECTION_SHA,'Projection mismatch: use the tested Torch environment'
    target=ASSETS/'projection.pt'
    if target.exists():assert torch.equal(torch.load(target,weights_only=True,map_location='cpu'),p)
    else:torch.save(p,target)
    print('Assets ready:',ASSETS,'projection tensor SHA256:',sha)

if __name__=='__main__':main()

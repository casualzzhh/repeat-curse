"""Forced-risk fixture checks mechanics only, never contributes evaluation outputs."""
from settings import save,ASSETS
from runtime import seed_all,require_gpu,load_backbone
from experiment import load_ar
from sampling import Collector
from repeats import spans
import torch

class ForcedRisk(torch.nn.Module):
    def forward(self,h,m):return torch.full((1,h.size(2),3),20.,device=h.device)
class DummyVerifier:
    def __init__(self):self.reset_called=False
    def reset(self):self.reset_called=True

def main():
    seed_all();require_gpu();model,tok,_=load_backbone();ar,ar_tok=load_ar()
    projection=torch.load(ASSETS/'projection.pt',weights_only=True).to('cuda')
    prompt='请介绍一种机械部件。'
    ids=tok.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,return_tensors='pt').to('cuda')
    prefix=ids.size(1);seq=torch.full((1,prefix+192),tok.mask_token_id,device='cuda',dtype=torch.long);seq[:,:prefix]=ids
    output=tok.encode('主轴、主轴、主轴、主轴、',add_special_tokens=False)
    seq[0,prefix:prefix+len(output)]=torch.tensor(output,device='cuda')
    assert any(s['kind']==1 for s in spans(output,tok))
    c=Collector(model,ar,tok,ar_tok,projection,prefix,prompt,ForcedRisk(),[.5]*3)
    for n in (1,2,3,4):
        f=seq.clone();f[:,prefix+n:]=tok.mask_token_id;c.observe(f,n)
    v=DummyVerifier();stats={'ar_model_calls':0,'rejection_correction_calls':0}
    with torch.inference_mode():rewound=c.after(seq,prefix+len(output)-1,4,v,stats)
    assert rewound is not None and v.reset_called and len(c.actions)==1
    action=c.actions[0]
    assert action['changed'] and action['ar_choice_committed'] and action['ar_fresh_prefix']
    assert int(seq[0,rewound])==action['new'] and seq[0,rewound+1:].eq(tok.mask_token_id).all()
    assert torch.equal(seq[:,:prefix],ids)
    save('forced-repair-smoke.json',dict(passed=True,fixture_only=True,action=action,prompt_unchanged=True,suffix_invalidated=True))
    print('REAL_MODEL_AR_REPAIR_FIXTURE_PASSED',flush=True)

if __name__=='__main__':main()

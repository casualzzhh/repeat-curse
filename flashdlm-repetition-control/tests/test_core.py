import hashlib,math,unittest
import torch
from risk import SpatiotemporalRisk
from backbone import LoRALinear,adapter_state,load_adapter,prediction_rows
from prepare_assets import PROJECTION_SHA,build_split

class CoreTests(unittest.TestCase):
    def test_three_independent_heads_and_four_frame_gradients(self):
        torch.manual_seed(42);head=SpatiotemporalRisk()
        history=torch.randn(1,4,16,128,requires_grad=True)
        logits=head(history,torch.ones(1,16,dtype=torch.bool))
        self.assertEqual(tuple(logits.shape),(1,16,3))
        torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.ones_like(logits)).backward()
        self.assertTrue(all(float(history.grad[:,i].norm())>0 for i in range(4)))
    def test_projection_exact(self):
        p=torch.randn(3584,128,generator=torch.Generator().manual_seed(501))/math.sqrt(3584)
        self.assertEqual(hashlib.sha256(p.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),PROJECTION_SHA)
    def test_lora_zero_initialization_and_base_frozen(self):
        base=torch.nn.Linear(7,5);layer=LoRALinear(base);x=torch.randn(2,7)
        self.assertTrue(torch.equal(layer(x),base(x)))
        layer(x).sum().backward();self.assertIsNone(base.weight.grad);self.assertGreater(float(layer.b.grad.norm()),0)
    def test_prediction_row_alignment(self):
        x=torch.arange(12).reshape(1,4,3)
        self.assertTrue(torch.equal(prediction_rows(x),x[:,[0,0,1,2]]))
    def test_data_split_reconstruction(self):
        class FakeTokenizer:
            eos_token_id=99
            def apply_chat_template(self,*args,**kwargs):return [1,2]
            def encode(self,*args,**kwargs):return [3,4]
        train=[{'question':f'q{i}','answer':'a'} for i in range(6)]
        test=[{'question':'unseen','answer':'a'}]
        a=build_split(train,test,FakeTokenizer(),ntrain=2,ndev=1,ntest=1)
        b=build_split(train,test,FakeTokenizer(),ntrain=2,ndev=1,ntest=1)
        self.assertEqual(a,b);self.assertEqual(a['train'][0]['ids'],[1,2,3,4,99])
        self.assertFalse({r['id'] for r in a['train']}&{r['id'] for r in a['dev']})

if __name__=='__main__':unittest.main()

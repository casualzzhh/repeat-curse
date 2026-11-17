"""Ordinary dense Dream training; no attention changes or runtime controls."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha/rank

    @property
    def weight(self): return self.base.weight

    def forward(self, x):
        return self.base(x)+(F.linear(F.linear(x.float(), self.a), self.b)*self.scale).to(x.dtype)


def install_lora(model):
    model.requires_grad_(False)
    for layer in model.model.layers:
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(layer.self_attn, name, LoRALinear(getattr(layer.self_attn, name)))


def adapter_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_adapter(model, state):
    params = dict(model.named_parameters())
    assert set(state) == {n for n, p in params.items() if p.requires_grad}
    with torch.no_grad():
        for n, value in state.items(): params[n].copy_(value)


def merge_lora(model):
    count = 0
    for layer in model.model.layers:
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            module = getattr(layer.self_attn, name)
            if isinstance(module, LoRALinear):
                with torch.no_grad(): module.base.weight.add_((module.b@module.a*module.scale).to(module.base.weight.dtype))
                setattr(layer.self_attn, name, module.base)
                count += 1
    return count


def prediction_rows(hidden):
    return torch.cat((hidden[:, :1], hidden[:, :-1]), dim=1)


def forward(model, ids, projection=None, prefix=0, checkpointed=False, middle_only=False):
    length = ids.size(1)
    for layer in model.model.layers:
        layer.self_attn.block_cache = None
        layer.self_attn.max_length = length
    hidden = model.model.embed_tokens(ids)
    positions = torch.arange(length, device=ids.device)[None]
    embeddings = model.model.rotary_emb(hidden, positions)
    feature = None
    for index, layer in enumerate(model.model.layers):
        def run(h, active=layer):
            # Checkpoint recomputation can interleave four different input lengths.
            # Restore this graph's metadata on every invocation, not only forward entry.
            active.self_attn.block_cache = None
            active.self_attn.max_length = length
            return active(h, attention_mask=None, position_ids=positions, use_block_diffusion=False,
                          use_cache=False, position_embeddings=embeddings)[0]
        hidden = checkpoint(run, hidden, use_reentrant=False) if checkpointed and torch.is_grad_enabled() else run(hidden)
        if index+1 == len(model.model.layers)//2:
            if projection is not None:
                feature = F.layer_norm(prediction_rows(hidden).float()@projection, (projection.size(1),))[:, prefix:]
            if middle_only: return None, feature
    return model.model.norm(hidden), feature

"""Small compatibility bridge containing only the APIs used by this experiment.

Model math and the native sampler are supplied by the attributed vendored runtime.
"""
from pathlib import Path
import os
from settings import option,OFFLINE
from huggingface_hub import snapshot_download

def resolve_model(name,revision):
    p=Path(name).expanduser()
    if p.is_dir():return str(p.resolve())
    return snapshot_download(name,revision=revision,local_files_only=OFFLINE)

DREAM=resolve_model(str(option('dream_model','Dream-org/Dream-v0-Instruct-7B')),
    str(option('dream_revision','05334cb9faaf763692dcf9d8737c642be2b2a6ae')))
QWEN=resolve_model(str(option('ar_model','Qwen/Qwen2.5-1.5B-Instruct')),
    str(option('ar_revision','989aa7980e4cf806f80c7fef2b1adb7bc71aa306')))
os.environ['REP_RESOLVED_DREAM_MODEL']=DREAM
os.environ['REP_OFFLINE']='1' if OFFLINE else '0'

from src.model.dream_flash.configuration_dream import ODreamConfig
from src.model.dream_flash.modeling_dream import DreamForCausalLM
from guided_diffusion.guided_diff_utils import AssistedDiffusionConfig

def reset(model,length,cache):
    model.model.cache_initialized=True
    for layer in model.model.layers:
        if cache:layer.reset_block_cache(1,length,32)
        else:
            layer.self_attn.block_cache=None
            layer.self_attn.max_length=length

def call(model,x,start,cache,freeze_at=None):
    import torch
    positions=torch.arange(start,start+x.size(1),device=x.device)[None]
    return model(x,None,positions,use_cache=False,use_block_diffusion=cache,
        use_full_query_attn=False,max_length=model._experiment_length,block_size=32,
        save_cache=cache,clean_idx=start if freeze_at is None else freeze_at,return_dict=True)

from settings import ROOT, ASSETS, SEED, MIN_FREE_GIB, save, digest
import gc
import hashlib
import json
import random
import torch
from transformers import AutoTokenizer, PreTrainedModel
import flash_experiment as flash


def seed_all(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def require_gpu():
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for model experiments; CPU unit tests do not require it.')
    if torch.cuda.mem_get_info()[0] < MIN_FREE_GIB*1024**3: raise RuntimeError('Insufficient free GPU memory; no other job interrupted')


def load_backbone():
    config = flash.ODreamConfig.from_pretrained(flash.DREAM, local_files_only=True)
    if config.hidden_size != 3584:raise ValueError('This reproduction targets the 3584-dimensional Dream-v0-Instruct-7B backbone.')
    config._attn_implementation = 'sdpa'
    model, info = PreTrainedModel.from_pretrained.__func__(flash.DreamForCausalLM, flash.DREAM, config=config,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True, output_loading_info=True)
    assert not info['missing_keys'] and not info['unexpected_keys'] and not info.get('mismatched_keys'), info
    model = model.to('cuda').eval()
    model.model.cache_initialized = True
    tokenizer = AutoTokenizer.from_pretrained(flash.DREAM, trust_remote_code=True, local_files_only=True)
    return model, tokenizer, info


def clear(model):
    for layer in model.model.layers: layer.self_attn.block_cache = None
    model.model.cache_initialized = True
    gc.collect()
    torch.cuda.empty_cache()


def state_digest(state):
    h = hashlib.sha256()
    for n, value in sorted(state.items()):
        h.update(n.encode()); h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def data():
    result = json.loads((ASSETS/'gsm8k_split.json').read_text(encoding='utf-8'))
    groups = [{r['id'] for r in result[k]} for k in ('train', 'dev', 'test')]
    assert len(groups[0]) == 1024 and len(groups[1]) == len(groups[2]) == 64
    assert not groups[0]&groups[1] and not (groups[0]|groups[1])&groups[2]
    save('data-provenance.json', {'data_path': str(ASSETS/'gsm8k_split.json'), 'sha256': digest(ASSETS/'gsm8k_split.json'),
        'source_hashes': result['source_hashes'], 'train': 1024, 'dev': 64, 'test': 64,
        'disjoint': True, 'train_truncated': sum(r['truncated'] for r in result['train']),
        'mask_only_main_objective': True, 'teacher_feature_cache_used': False})
    return result

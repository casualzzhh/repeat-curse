"""Portable paths for the screenshot V9 pilot. No credentials or host-specific paths."""
import hashlib,json,os,sys,time
from pathlib import Path

CODE=Path(__file__).resolve().parent
config_file=os.environ.get('REP_CONFIG')
CONFIG=json.loads(Path(config_file).read_text(encoding='utf-8')) if config_file else {}

def option(name,default):
    return os.environ.get('REP_'+name.upper(),CONFIG.get(name,default))

def path_option(name,default):
    p=Path(option(name,default)).expanduser()
    return (p if p.is_absolute() else CODE/p).resolve()

ROOT=path_option('run_dir','runs/pilot')
ASSETS=path_option('asset_dir','data')
FLASH=path_option('flash_dir','third_party/flash_dlm')
ROOT.mkdir(parents=True,exist_ok=True)
SEED=42
OFFLINE=str(option('offline','1')).lower() not in ('0','false','no')
MIN_FREE_GIB=float(option('min_free_gib',22))
os.environ['HF_HUB_OFFLINE']='1' if OFFLINE else '0'
os.environ['TRANSFORMERS_OFFLINE']='1' if OFFLINE else '0'
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
sys.path.insert(0,str(FLASH))

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def save(name,obj):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');q.replace(p)

def log(event,**kw):
    d=dict(event=event,time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw);save('status.json',d)
    with (ROOT/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
    print(json.dumps(d,ensure_ascii=False),flush=True)

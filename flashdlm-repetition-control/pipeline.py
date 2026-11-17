"""Run the complete screenshot experiment in an isolated output directory."""
from settings import CODE,ROOT,ASSETS,save,log,digest,option
import argparse,json,os,shutil,subprocess,sys,time,traceback

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume',action='store_true',help='Resume completed stages; not an in-training optimizer resume')
    args=p.parse_args();lock=ROOT/'pipeline.lock'
    if lock.exists():raise RuntimeError(f'Run lock exists: {lock}. Check the recorded process before removing a stale lock.')
    if (ROOT/'checkpoint.pt').exists() and not (ROOT/'training-complete.json').exists():
        raise RuntimeError('Partial training checkpoint has no optimizer state. Choose a fresh REP_RUN_DIR to retrain.')
    if (ROOT/'launcher.json').exists() and not args.resume:raise RuntimeError('Output directory already used; choose REP_RUN_DIR or --resume.')
    sources=list(CODE.glob('*.py'))+list((CODE/'third_party').rglob('*.py'))
    signature=dict(code={str(p.relative_to(CODE)):digest(p) for p in sources},
        assets={n:digest(ASSETS/n) for n in ('gsm8k_split.json','evaluation-prompts.json')},
        models={name:str(option(name,default)) for name,default in [('dream_model','Dream-org/Dream-v0-Instruct-7B'),
            ('dream_revision','05334cb9faaf763692dcf9d8737c642be2b2a6ae'),('ar_model','Qwen/Qwen2.5-1.5B-Instruct'),
            ('ar_revision','989aa7980e4cf806f80c7fef2b1adb7bc71aa306')]})
    prior=ROOT/'pipeline-sources.json'
    if args.resume and prior.exists():
        if json.loads(prior.read_text(encoding='utf-8'))!=signature:raise RuntimeError('Code, model settings or assets changed; choose a fresh run directory instead of mixing stages.')
    else:save('pipeline-sources.json',signature)
    lock.write_text(json.dumps({'pid':os.getpid(),'time':time.time()}),encoding='utf-8')
    os.environ['REP_RUN_DIR']=str(ROOT);os.environ['REP_ASSET_DIR']=str(ASSETS)
    save('launcher.json',dict(pid=os.getpid(),started=time.time(),code=str(CODE)))
    if (ROOT/'failed.json').exists():
        old=ROOT/'failure-history';old.mkdir(exist_ok=True)
        (ROOT/'failed.json').replace(old/f'{time.time_ns()}.json')
    snapshot=ROOT/'code';snapshot.mkdir(exist_ok=True)
    for source in CODE.glob('*.py'):shutil.copy2(source,snapshot/source.name)
    stages=[('assets',['prepare_assets.py'],None),('tests',['-m','unittest','discover','-s','tests','-v'],None),
        ('smoke',['smoke.py'],'forced-repair-smoke.json'),('train',['experiment.py','train'],'training-complete.json'),
        ('calibrate',['experiment.py','calibrate'],'thresholds.json'),('original',['experiment.py','original'],'eval-original-complete.json'),
        ('complete',['experiment.py','complete'],'eval-complete-complete.json'),('report',['report.py'],None)]
    try:
        for name,command,marker in stages:
            if args.resume and marker and (ROOT/marker).exists():log('stage_skipped',stage=name);continue
            log('stage_start',stage=name)
            with (ROOT/(name+'.log')).open('w',encoding='utf-8') as f:
                subprocess.run([sys.executable,'-u']+command,cwd=CODE,stdout=f,stderr=subprocess.STDOUT,check=True)
            log('stage_complete',stage=name)
        save('complete.json',dict(ok=True,time=time.time()))
        subprocess.run([sys.executable,str(CODE/'report.py')],cwd=CODE,stdout=subprocess.DEVNULL,check=True)
        log('complete',ok=True)
    except Exception:
        save('failed.json',dict(error=traceback.format_exc(),time=time.time()));raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()

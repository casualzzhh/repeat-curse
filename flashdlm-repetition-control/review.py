"""Create human review templates and reject annotations for changed output texts."""
import argparse,hashlib,json
from settings import ROOT,save

def text_sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('init','summarize'));args=parser.parse_args()
    cases=json.loads((ROOT/'paired-cases.json').read_text(encoding='utf-8'));path=ROOT/'manual-review.json'
    if args.command=='init':
        if path.exists():raise FileExistsError('Review file exists; refusing to overwrite human annotations')
        save('manual-review.json',dict(blind=False,rubric='At least3 meaningless consecutive phrase copies,2 substantially same clauses without new information,or extended meaningless character runs; exclude requested repetition and ordinary digits; null means uncertain.',
            rows=[dict(id=c['id'],kind=c['kind'],original_sha256=text_sha(c['original']),complete_sha256=text_sha(c['complete']),
                original_clear_loop=None,complete_clear_loop=None,notes='') for c in cases]))
        print('Fill original_clear_loop and complete_clear_loop with true/false; null remains uncertain.');return
    labels=json.loads(path.read_text(encoding='utf-8'));by={r['id']:r for r in labels['rows']};assert len(by)==len(cases)
    evaluated=[];uncertain=[]
    for c in cases:
        r=by[c['id']]
        assert r['original_sha256']==text_sha(c['original']) and r['complete_sha256']==text_sha(c['complete']),'Output changed after annotation'
        if c['kind']=='intent':continue
        values=[r['original_clear_loop'],r['complete_clear_loop']]
        assert all(v is None or isinstance(v,bool) for v in values)
        if None in values:uncertain.append(c['id'])
        else:evaluated.append(r)
    result=dict(blind=labels.get('blind',False),evaluated=len(evaluated),uncertain=uncertain,
        original_clear_loops=sum(r['original_clear_loop'] for r in evaluated),complete_clear_loops=sum(r['complete_clear_loop'] for r in evaluated),
        resolved=sum(r['original_clear_loop'] and not r['complete_clear_loop'] for r in evaluated),
        introduced=sum(not r['original_clear_loop'] and r['complete_clear_loop'] for r in evaluated))
    save('manual-summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

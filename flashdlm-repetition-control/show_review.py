from settings import ROOT
import json,sys
cases=json.loads((ROOT/'paired-cases.json').read_text(encoding='utf-8'))
start=int(sys.argv[1]);end=min(len(cases),start+int(sys.argv[2]))
for i in range(start,end):
    c=cases[i]
    print(f'CASE {i} {c["id"]} ({c["kind"]})')
    print('PROMPT:',c['prompt'])
    print('METRICS:',json.dumps(dict(original=c['original_metrics']['spans'],complete=c['complete_metrics']['spans'],
        original_correct=c['original_correct'],complete_correct=c['complete_correct']),ensure_ascii=False))
    print('ORIGINAL:',c['original'])
    print('COMPLETE:',c['complete'])
    print('ACTIONS:',json.dumps(c['actions'],ensure_ascii=False))
    print('ENDCASE',i,flush=True)

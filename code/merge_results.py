"""Merge disjoint scenario runs without copying raw traces; final manifest records paths."""
import json
from pathlib import Path
from run_experiments import summarize

root=Path(__file__).resolve().parents[1]/'results'
if (root/'manifest.json').exists():
    current=json.loads((root/'manifest.json').read_text())
    if current.get('cases') and all(not x.get('path') for x in current['cases'].values()):
        print('Results already consolidated; no merge is needed.')
        raise SystemExit(0)
rows=[];man={};paired=[]
for sub in ['core','sensitivity','extra','feedback','frequencycheck']:
    d=root/sub
    rows+=[r for r in json.loads((d/'per_seed.json').read_text()) if not (sub=='sensitivity' and r['case']=='frequency4')]
    m=json.loads((d/'manifest.json').read_text())
    for name,info in m['cases'].items():
        if sub=='sensitivity' and name=='frequency4':continue
        assert name not in man
        man[name]=dict(**info,path=sub)
    paired+=[r for r in json.loads((d/'paired.json').read_text()) if not (sub=='sensitivity' and r['case']=='frequency4')]
(root/'summary.json').write_text(json.dumps(summarize(rows),indent=2))
(root/'per_seed.json').write_text(json.dumps(rows,indent=2))
(root/'paired.json').write_text(json.dumps(paired,indent=2))
(root/'manifest.json').write_text(json.dumps(dict(cases=man,seeds=list(range(10))),indent=2))
print(len(man),'scenarios',len(rows),'seed-method evaluations')

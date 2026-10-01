"""Extend Union-Frozen to gradual shift and external-interference mismatch."""
import json
from concurrent.futures import ProcessPoolExecutor
from reviewer_experiments import worker,ROOT
from run_experiments import summarize

def main():
 out=ROOT/'results/reviewer'
 jobs=[(case,s) for case in ['union_gradual','union_interference'] for s in range(10)]
 with ProcessPoolExecutor(max_workers=8) as pool:groups=list(pool.map(worker,jobs))
 additions=[r for group in groups for r in group]
 rows=json.loads((out/'per_seed.json').read_text());rows=[r for r in rows if r['case'] not in ['union_gradual','union_interference']]+additions
 (out/'per_seed.json').write_text(json.dumps(rows,indent=2));(out/'summary.json').write_text(json.dumps(summarize(rows),indent=2))
 (out/'union_extension_config.json').write_text(json.dumps(dict(jobs=jobs,new_decisions=60000,description='Same default settings and namespaces as original gradual/interference experiments'),indent=2))
 print(json.dumps(summarize(additions),indent=2))
if __name__=='__main__':main()

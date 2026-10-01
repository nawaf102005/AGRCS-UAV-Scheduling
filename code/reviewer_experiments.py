"""Additional review experiments. Fixed configurations; no test-based tuning.

Run with --workers 8. Each seed has disjoint development/calibration/evaluation
namespaces matching the original experiments. Existing seed 0--9 results are
retained for the 30-seed contrast; seeds 10--29 are fresh evaluations.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,json
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from agrcs.model import Config,Channel,ScaleModel
from agrcs.calibration import calibrate,Controller
from agrcs.scheduler import Policy
from run_experiments import summarize
ROOT=Path(__file__).resolve().parents[1]

class UnionFrozen(Policy):
    """Frozen Bonferroni graph family, common assignment/power/overhead model."""
    def __init__(self,c,scale,cal):
        super().__init__(c,'Frozen-GRCS',scale,cal)
        self.degrees=c.degrees
        cc=replace(c,alpha=c.alpha/len(c.degrees))
        self.controllers={d:Controller(cal[d],cc,'frozen') for d in self.degrees}

def worker(job):
    case,seed=job;c=replace(Config(),scenario=(case.removeprefix('union_') if case.startswith('union_') else 'abrupt'))
    scale=ScaleModel().fit(c,100000+seed);cal=calibrate(c,scale,200000+seed)
    if case=='alpha':
        specs=[(f'{m}@{a}',replace(c,alpha=a),m) for a in [.05,.2] for m in ['A-GRCS','A-GRCS-ACK']]
        specs+=[('Union-Frozen',c,'Union-Frozen')]
    elif case.startswith('union_'):specs=[('Union-Frozen',c,'Union-Frozen')]
    else:specs=[(m,c,m) for m in ['A-GRCS-ACK','Frozen-GRCS']]
    policies={name:(UnionFrozen(cc,scale,cal) if m=='Union-Frozen' else Policy(cc,m,scale,cal)) for name,cc,m in specs}
    env=Channel(c,300000+seed);records={name:[] for name in policies}
    for _ in range(c.slots):
        obs,true,ref=env.sample()
        for name,p in policies.items():records[name].append(p.feedback(obs,true,ref,p.schedule(obs)))
    out=ROOT/'results/reviewer';out.mkdir(exist_ok=True);rows=[]
    for name,p in policies.items():
        raw={k:np.asarray([x[k] for x in records[name]]) for k in ['goodput','failure','miscoverage','rf','idle','queue','degree','alpha_state']}
        np.savez_compressed(out/f'{case}_{seed}_{name}.npz',**raw)
        if p.method in ['A-GRCS','A-GRCS-ACK']:
            key='failure' if p.method=='A-GRCS-ACK' else 'miscoverage';h=np.arange(1,c.slots+1);D=len(p.degrees);a=p.c.alpha
            residual=np.max(abs(raw[key].cumsum()-(a*h+(D*a-raw['alpha_state'])/c.gamma)))
            assert residual<1e-7
            assert np.all(raw[key].cumsum()/h<=a+D*(a+c.gamma*(1-a))/(c.gamma*h)+1e-12)
        rows.append(dict(case=case,method=name,seed=seed,**{k:float(v.mean()) for k,v in raw.items()}))
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    jobs=[('alpha',s) for s in range(10)]+[('union_stationary',s) for s in range(10)]+[('extra_seeds',s) for s in range(10,30)]
    out=ROOT/'results/reviewer';out.mkdir(exist_ok=True)
    (out/'config.json').write_text(json.dumps(dict(base=replace(Config(),scenario='abrupt').dict(),jobs=jobs,alpha_grid=[.05,.1,.2],new_decisions=300000,description='Additional fixed-config review experiments; alpha=0.1 uses retained original seeds 0--9'),indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(worker,j):j for j in jobs}
        for f in as_completed(futures):
            rows+=f.result();print('Completed',futures[f],flush=True)
            (out/'per_seed.json').write_text(json.dumps(sorted(rows,key=lambda r:(r['case'],r['method'],r['seed'])),indent=2))
    (out/'summary.json').write_text(json.dumps(summarize(rows),indent=2))
if __name__=='__main__':main()

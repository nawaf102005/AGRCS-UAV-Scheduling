"""Finite-backlog service phase: scheduling affects completion time and propulsion.

Energy includes steady-flight propulsion until payload completion, not takeoff,
return-to-base, landing, battery aging, or propulsion-trajectory optimization.
"""
import json,argparse
from pathlib import Path
from dataclasses import replace
import numpy as np
from agrcs.model import Config,Channel,ScaleModel
from agrcs.calibration import calibrate
from agrcs.scheduler import Policy
from agrcs.rl import Network
from run_experiments import summarize

def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=root/'results')
    out=parser.parse_args().out;out.mkdir(parents=True,exist_ok=True)
    c=replace(Config(),initial_queue=20.,arrival=0.,scenario='stationary',slots=600)
    methods=['A-GRCS','Frozen-GRCS','Rolling-GRCS','ACI-d2','ACI-Full','Nominal','PF-Power','DDQN-Match','Oracle']
    rows=[]
    for seed in range(10):
        scale=ScaleModel().fit(c,100000+seed);cal=calibrate(c,scale,200000+seed)
        net=Network.load(root/'models'/f'ddqn_seed{seed%3}.npz')
        policies={m:Policy(c,m,scale,cal,net) for m in methods}
        accum={m:dict(time=0.,rf_energy=0.,propulsion_energy=0.,failed_slots=0.) for m in methods}
        done={m:False for m in methods};env=Channel(c,400000+seed)
        for t in range(c.slots):
            obs,true,ref=env.sample()
            for m,p in policies.items():
                if done[m]:continue
                a=p.schedule(obs,oracle=true if m=='Oracle' else None)
                metric=p.feedback(obs,true,ref,a);z=accum[m]
                z['time']+=c.slot;z['rf_energy']+=metric['rf']
                z['propulsion_energy']+=metric['mission']-metric['rf'];z['failed_slots']+=metric['failure']
                done[m]=p.queue.max()<=.01*c.initial_queue
            if all(done.values()):break
        for m in methods:
            z=accum[m];z['total_energy']=z['rf_energy']+z['propulsion_energy']
            rows.append(dict(case='bulk',method=m,seed=seed,completed=int(done[m]),
                             remaining=float(policies[m].queue.sum()),**z))
        print('bulk seed',seed,'completed',sum(done.values()),'of',len(methods),flush=True)
    (out/'bulk_per_seed.json').write_text(json.dumps(rows,indent=2))
    (out/'bulk_summary.json').write_text(json.dumps(summarize(rows),indent=2))
    (out/'bulk_config.json').write_text(json.dumps(c.dict(),indent=2))
if __name__=='__main__':main()

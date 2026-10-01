"""Serial timing on cached observations, isolated from full experiment workers."""
import json,time,argparse,platform,sys,os
import scipy
from dataclasses import replace
from pathlib import Path
import numpy as np
from agrcs.model import Config,Channel,ScaleModel
from agrcs.calibration import calibrate
from agrcs.scheduler import Policy
from agrcs.rl import Network

def main():
    root=Path(__file__).resolve().parents[1];rows=[];raw={}
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=root/'results')
    out=parser.parse_args().out;out.mkdir(parents=True,exist_ok=True)
    for K in [24,50,100]:
        c=replace(Config(),users=K,train_n=100,cal_n=200)
        scale=ScaleModel().fit(c,770001);cal=calibrate(c,scale,770002)
        net=Network.load(root/'models'/'ddqn_seed0.npz')
        env=Channel(c,770003);samples=[env.sample() for _ in range(600)]
        for method in ['A-GRCS','ACI-d2','ACI-Full','PF-Power','DDQN-Match']:
            p=Policy(c,method,scale,cal,net);ts=[]
            for i,(obs,true,ref) in enumerate(samples):
                start=time.perf_counter_ns();a=p.schedule(obs)
                elapsed=(time.perf_counter_ns()-start)/1e6
                if i>=100:ts.append(elapsed)
                p.feedback(obs,true,ref,a)
            raw[f'{K}_{method}']=np.array(ts)
            rows.append(dict(users=K,method=method,mean_ms=float(np.mean(ts)),p95_ms=float(np.quantile(ts,.95)),max_ms=float(np.max(ts)),deadline_misses=int(np.sum(np.array(ts)>100))))
    (out/'isolated_latency.json').write_text(json.dumps(rows,indent=2))
    np.savez_compressed(out/'isolated_latency_raw.npz',**raw)
    cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor())
    (out/'latency_environment.json').write_text(json.dumps(dict(cpu=cpu,platform=platform.platform(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,warmup=100,timed_calls=500,threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}),indent=2))
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()

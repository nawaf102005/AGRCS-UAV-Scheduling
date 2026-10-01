"""Run independent seeded trajectories, paired policies, and retain auditable traces."""
import argparse,json,csv,sys,platform,time,hashlib
from pathlib import Path
from dataclasses import replace
import numpy as np
import scipy
from scipy.stats import t as student_t
from agrcs.model import Config,Channel,ScaleModel
from agrcs.calibration import calibrate
from agrcs.scheduler import Policy,METHODS
from agrcs.rl import Network

def summarize(rows):
    out=[];groups={}
    for r in rows:groups.setdefault((r['case'],r['method']),[]).append(r)
    for (case,method),rs in groups.items():
        row=dict(case=case,method=method,n=len(rs))
        for key in rs[0]:
            if key in ('case','method','seed'):continue
            x=np.array([r[key] for r in rs],float)
            row[key]=float(x.mean())
            row[key+'_ci']=float(student_t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else 0.
        out.append(row)
    return out

def run(c,seed,methods,out,name,models):
    scale=ScaleModel().fit(c,100000+seed)
    cal=calibrate(c,scale,200000+seed)
    np.savez_compressed(out/'calibration'/f'{name}_{seed}.npz',beta=scale.beta,**{f'd{d}':v for d,v in cal.items()})
    env=Channel(c,300000+seed)
    net=Network.load(models/f'ddqn_seed{seed%3}.npz') if 'DDQN-Match' in methods else None
    policies={m:Policy(c,m,scale,cal,net) for m in methods};traces={m:[] for m in methods}
    for t in range(c.slots):
        obs,true,ref=env.sample()
        for m in methods[t%len(methods):]+methods[:t%len(methods)]:
            p=policies[m];start=time.perf_counter_ns()
            act=p.schedule(obs,oracle=true if m=='Oracle' else None)
            latency=(time.perf_counter_ns()-start)/1e6
            metric=p.feedback(obs,true,ref,act);metric['latency']=latency
            traces[m].append(metric)
    rows=[]
    for m in methods:
        raw={k:np.asarray([r[k] for r in traces[m]]) for k in traces[m][0]}
        np.savez_compressed(out/'traces'/f'{name}_{seed}_{m}.npz',**raw)
        r=dict(case=name,method=m,seed=seed)
        for key,v in raw.items():
            if key not in ['delivered','margin']:r[key]=float(v.mean())
        r['margin_finite']=float(np.mean(raw['margin'][np.isfinite(raw['margin'])]))
        r['infinite_margin']=float(np.mean(np.isposinf(raw['margin'])))
        r['post_failure']=float(raw['failure'][int(.5*c.slots):].mean())
        r['post_goodput']=float(raw['goodput'][int(.5*c.slots):].mean())
        r['post_miscoverage']=float(raw['miscoverage'][int(.5*c.slots):].mean())
        r['p95_latency']=float(np.quantile(raw['latency'],.95))
        delivered=raw['delivered'].sum(0)
        r['jain']=float(delivered.sum()**2/(c.users*(delivered**2).sum()+1e-20))
        r['final_queue']=float(policies[m].queue.mean())
        if m in ('A-GRCS','A-GRCS-Budget','A-GRCS-ACK','ACI-d2','ACI-Full','PF-Power'):
            D=len(policies[m].controllers)
            bound=c.alpha+D*(c.alpha+c.gamma*(1-c.alpha))/(c.gamma*c.slots)
            risk_key='failure' if m=='A-GRCS-ACK' else 'miscoverage'
            err=raw[risk_key].sum()
            identity=c.alpha*c.slots+(D*c.alpha-sum(x.a for x in policies[m].controllers.values()))/c.gamma
            assert abs(err-identity)<1e-7, (m,err,identity)
            assert r[risk_key]<=bound+1e-12
        rows.append(r)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',choices=['quick','paper'],default='quick')
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    p.add_argument('--models',type=Path,default=Path(__file__).resolve().parents[1]/'models')
    p.add_argument('--cases',nargs='*');p.add_argument('--seeds',type=int)
    a=p.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=True)
    old_manifest=json.loads((out/'manifest.json').read_text()) if (out/'manifest.json').exists() else {}
    for x in ['traces','calibration']:(out/x).mkdir(exist_ok=True)
    c=Config() if a.profile=='paper' else replace(Config(),slots=120,train_n=40,cal_n=100)
    n=a.seeds or (10 if a.profile=='paper' else 2)
    compact=['A-GRCS','Frozen-GRCS','Rolling-GRCS','ACI-d2','ACI-Full','Nominal','PF-Power']
    methods=METHODS if (a.models/'ddqn_seed2.npz').exists() else [m for m in METHODS if m!='DDQN-Match']
    cases=[('stationary',c,methods),('abrupt',replace(c,scenario='abrupt'),methods)]
    if a.profile=='paper':
        cases += [(s,replace(c,scenario=s),compact) for s in ['gradual','recurrent','interference']]
        cases += [(f'users_{k}',replace(c,users=k,scenario='abrupt'),compact) for k in [50,100]]
        cases += [(f'eta_{eta}',replace(c,eta=eta,scenario='abrupt'),['A-GRCS','ACI-d2','ACI-Full']) for eta in [.1,2.,10.]]
        cases += [(f'gamma_{g}',replace(c,gamma=g,scenario='abrupt'),['A-GRCS','ACI-d2']) for g in [.005,.05]]
        cases += [('frequency4',replace(c,rbs=4,powers=(.025,.05,.1,.25),scenario='abrupt'),compact),
                  ('correlation95',replace(c,temporal=.95,error_temporal=.95,scenario='abrupt'),compact),
                  ('load_high',replace(c,arrival=.36,scenario='abrupt'),methods),
                  ('budget',replace(c,scenario='abrupt'),['A-GRCS','A-GRCS-Budget'])]
        cases += [('unit_scale',replace(c,scale_mode='unit',scenario='abrupt'),['A-GRCS','Frozen-GRCS','ACI-d2']),
                  ('reference2',replace(c,reference_error=2.,scenario='abrupt'),['A-GRCS','Frozen-GRCS','ACI-d2'])]
        cases += [('no_overhead',replace(c,reference_fraction=0.,scenario='abrupt'),['A-GRCS','ACI-d2','ACI-Full']),
                  ('ack_abrupt',replace(c,scenario='abrupt'),['A-GRCS','A-GRCS-ACK','Frozen-GRCS','DDQN-Match']),
                  ('ack_interference',replace(c,scenario='interference'),['A-GRCS','A-GRCS-ACK','Frozen-GRCS']),
                  ('ack_load',replace(c,scenario='abrupt',arrival=.36),['A-GRCS','A-GRCS-ACK','Frozen-GRCS','DDQN-Match'])]
    if a.cases:cases=[x for x in cases if x[0] in a.cases]
    manifest=dict(profile=a.profile,seeds=list(range(n)),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
                  platform=platform.platform(),cases={name:dict(config=cc.dict(),methods=ms) for name,cc,ms in cases})
    manifest['source_sha256']={str(f.relative_to(Path(__file__).parent)):hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in sorted(Path(__file__).parent.rglob('*.py'))}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[]
    for name,cc,ms in cases:
        start=time.time()
        existing=out/f'case_{name}.json'
        same_config=json.dumps(old_manifest.get('cases',{}).get(name),sort_keys=True)==json.dumps(manifest['cases'][name],sort_keys=True)
        same_source=old_manifest.get('source_sha256')==manifest['source_sha256']
        if existing.exists() and same_config and same_source:
            rr=json.loads(existing.read_text())
            if len(rr)==n*len(ms):rows+=rr;print(name,'resumed',flush=True);continue
        rr=[]
        for seed in range(n):
            rr+=run(cc,seed,ms,out,name,a.models)
            print(name,'seed',seed,'done',flush=True)
        existing.write_text(json.dumps(rr,indent=2));rows+=rr
        (out/'summary.json').write_text(json.dumps(summarize(rows),indent=2))
        print(name,round(time.time()-start,1),'seconds',flush=True)
    (out/'summary.json').write_text(json.dumps(summarize(rows),indent=2))
    (out/'per_seed.json').write_text(json.dumps(rows,indent=2))
    if rows:
        with (out/'per_seed.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    paired=[]
    for name,_,ms in cases:
        aa={r['seed']:r for r in rows if r['case']==name and r['method']=='A-GRCS'}
        for m in ms:
            if m=='A-GRCS':continue
            bb={r['seed']:r for r in rows if r['case']==name and r['method']==m}
            for key in ['failure','goodput','rf','queue','degree']:
                vals=np.array([aa[s][key]-bb[s][key] for s in aa])
                paired.append(dict(case=name,baseline=m,metric=key,difference=float(vals.mean()),
                    ci95=float(student_t.ppf(.975,len(vals)-1)*vals.std(ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else 0.))
    (out/'paired.json').write_text(json.dumps(paired,indent=2))
if __name__=='__main__':main()

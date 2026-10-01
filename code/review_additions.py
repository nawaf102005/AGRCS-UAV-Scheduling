"""Reproduce Gaussian-reference stress tests and confidence-interval audits.

Run after benchmark_latency.py and build_report.py. Reference development and
calibration retain bounded noise; evaluation uses independent 1-dB Gaussian noise.
All ACK horizon prefixes are verified. No Gaussian tail certificate is assumed.
"""
import argparse,json
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.stats import t as student_t
from agrcs.model import Config,Channel,ScaleModel
from agrcs.calibration import calibrate
from agrcs.scheduler import Policy
from run_experiments import summarize

def main():
 root=Path(__file__).resolve().parents[1]
 parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,default=root/'results');parser.add_argument('--audit-only',action='store_true',help='Audit retained seed summaries without simulation or manuscript changes');args=parser.parse_args();res=args.results
 out=res/'review_reference';out.mkdir(parents=True,exist_ok=True);rows=[];checks=[]
 if not args.audit_only:
  c=replace(Config(),scenario='abrupt')
  for seed in range(10):
   scale=ScaleModel().fit(c,100000+seed);cal=calibrate(c,scale,200000+seed)
   env=Channel(c,300000+seed);rng=np.random.default_rng(900000+seed)
   policies={m:Policy(c,m,scale,cal) for m in ['A-GRCS-ACK','Frozen-GRCS']};records={m:[] for m in policies}
   for _ in range(c.slots):
    obs,true,_=env.sample();ref=true+rng.normal(0,1.,size=true.shape)
    for m,p in policies.items():records[m].append(p.feedback(obs,true,ref,p.schedule(obs)))
   for m,rr in records.items():
    raw={k:np.asarray([r[k] for r in rr]) for k in rr[0]};np.savez_compressed(out/f'{seed}_{m}.npz',**raw)
    rows.append(dict(case='gaussian_reference',method=m,seed=seed,**{k:float(raw[k].mean()) for k in ['goodput','failure','rf','queue','idle']}))
    if m=='A-GRCS-ACK':
     h=np.arange(1,c.slots+1);D=len(c.degrees);identity=c.alpha*h+(D*c.alpha-raw['alpha_state'])/c.gamma
     gap=float(np.max(abs(raw['failure'].cumsum()-identity)));bound=c.alpha+D*(c.alpha+c.gamma*(1-c.alpha))/(c.gamma*h)
     assert gap<1e-7 and np.all(raw['failure'].cumsum()/h<=bound+1e-12)
     checks.append(dict(seed=seed,all_prefix_identity_max_error=gap))
   print('Gaussian-reference seed',seed,'complete',flush=True)
  summary=summarize(rows)
  for name,value in [('per_seed',rows),('summary',summary),('checks',checks),('config',dict(config=c.dict(),reference_sd_db=1.,development_reference='uniform +/-0.5 dB',evaluation_reference='independent Gaussian 1 dB',reference_rng_namespace=900000,seeds=list(range(10))))]:
   (out/f'{name}.json').write_text(json.dumps(value,indent=2))
 # Independent interval recomputation from seed records, with t9 and sample SD.
 audits=[]
 for source,target,group in [('per_seed.json','summary.json',('case','method')),('bulk_per_seed.json','bulk_summary.json',('method',)),('review_reference/per_seed.json','review_reference/summary.json',('case','method'))]:
  records=json.loads((res/source).read_text())
  for r in json.loads((res/target).read_text()):
   rr=[x for x in records if all(x[g]==r[g] for g in group)]
   assert len(rr)==10 and len({x['seed'] for x in rr})==10
   for key,val in r.items():
    if not key.endswith('_ci'):continue
    metric=key[:-3];v=np.array([x[metric] for x in rr]);half=student_t.ppf(.975,9)*v.std(ddof=1)/np.sqrt(10)
    assert np.isclose(v.mean(),r[metric],rtol=1e-12,atol=1e-12) and np.isclose(half,val,rtol=1e-12,atol=1e-12)
    audits.append(dict(source=target,metric=metric,group={g:r[g] for g in group},maximum_error=float(max(abs(v.mean()-r[metric]),abs(half-val)))))
 records=json.loads((res/'per_seed.json').read_text())
 for file in ['paired.json','ack_paired.json']:
  for r in json.loads((res/file).read_text()):
   case=r.get('case','ack_abrupt');m='A-GRCS-ACK' if file=='ack_paired.json' else 'A-GRCS'
   aa={x['seed']:x for x in records if x['case']==case and x['method']==m};bb={x['seed']:x for x in records if x['case']==case and x['method']==r['baseline']}
   v=np.array([aa[i][r['metric']]-bb[i][r['metric']] for i in range(10)]);half=student_t.ppf(.975,9)*v.std(ddof=1)/np.sqrt(10)
   assert np.isclose(v.mean(),r['difference'],atol=1e-12,rtol=1e-12) and np.isclose(half,r['ci95'],atol=1e-12,rtol=1e-12)
   audits.append(dict(source=file,metric=r['metric'],maximum_error=float(max(abs(v.mean()-r['difference']),abs(half-r['ci95'])))))
 audit=dict(intervals_checked=len(audits),maximum_error=max(x['maximum_error'] for x in audits),definition='two-sided pointwise 95% Student t; ten independent seed means; ddof=1',checks=audits)
 (root/'docs/REVIEW_STATISTICS_AUDIT.json').write_text(json.dumps(audit,indent=2));print('Audited intervals:',len(audits),'maximum error:',audit['maximum_error'],flush=True)
 if args.audit_only:return
 print(json.dumps({'review_reference_summary': json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else [], 'statistics_audit': audit},indent=2))
if __name__=='__main__':main()

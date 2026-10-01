"""Independent mathematical and implementation checks, including adverse sequences."""
import itertools,json
from pathlib import Path
from dataclasses import replace
import numpy as np
from agrcs.model import Config,Channel,ScaleModel,graph,rate
from agrcs.scheduler import assign,Policy
from agrcs.calibration import Controller,quantile,calibrate
from agrcs.rl import Network

def exhaustive():
    c=replace(Config(),users=3,uavs=2,rbs=2,degrees=(1,2),powers=(.05,.5))
    rng=np.random.default_rng(9123);max_gap=0.
    for seed in range(12):
        obs,true,ref=Channel(c,seed).sample();s=np.ones_like(obs.estimate)*4
        queue=rng.uniform(.01,2,c.users);previous=np.array([-1,0,1]);av=np.ones(c.users)
        threshold=2.;a=assign(obs,s,queue,previous,av,c,2,threshold)
        best=0.;opts=[None]+list(itertools.product(range(2),range(2),c.powers))
        for actions in itertools.product(opts,repeat=c.users):
            used=[(z[0],z[1]) for z in actions if z is not None]
            if len(used)!=len(set(used)):continue
            val=0.
            for k,z in enumerate(actions):
                if z is None:continue
                u,r,p=z;sw=int(previous[k]>=0 and previous[k]!=u)
                duration=c.slot*(1-c.reference_fraction*2)*(1-.1*sw)
                b=min(queue[k],duration*rate(obs.estimate[k,u,r]-threshold*s[k,u,r],p,c))
                val+=(1+min(queue[k]/2,20))*b-c.eta*c.slot*p-c.handoff_cost*sw
            best=max(best,val)
        max_gap=max(max_gap,abs(a['utility']-best))
    assert max_gap<1e-9
    return max_gap

def arbitrary_sequence():
    c=Config();rng=np.random.default_rng(44)
    ctr={d:Controller(rng.normal(size=1000),c) for d in c.degrees}
    E=0;T=20000
    for t in range(T):
        d=c.degrees[t%3] if t%7 else max(ctr,key=lambda k:ctr[k].a)
        q=ctr[d].threshold()
        # Strongly nonstationary and deliberately adversarial post-selection scores.
        z=(q+1 if np.isfinite(q) else rng.normal()) if t%2 else 10*np.sin(t/19)+rng.standard_t(2)
        E+=ctr[d].update(z,q)
    rhs=c.alpha*T+(len(ctr)*c.alpha-sum(x.a for x in ctr.values()))/c.gamma
    upper=c.alpha+len(ctr)*(c.alpha+c.gamma*(1-c.alpha))/(c.gamma*T)
    assert abs(E-rhs)<1e-7 and E/T<=upper
    return dict(error_rate=E/T,upper_bound=upper,telescoping_gap=abs(E-rhs))

def interfaces_and_feedback():
    c=replace(Config(),slots=100,train_n=30,cal_n=80,scenario='abrupt')
    scale=ScaleModel().fit(c,101);cal=calibrate(c,scale,102);env=Channel(c,103)
    p=Policy(c,'A-GRCS',scale,cal)
    for _ in range(100):
        obs,true,ref=env.sample()
        assert not hasattr(obs,'true') and not hasattr(obs,'sigma')
        assert np.all(ref-c.reference_error<=true+1e-10)
        a=p.schedule(obs);p.feedback(obs,true,ref,a)
    assert quantile([1,2,3],0)==np.inf and quantile([1,2,3],1)==-np.inf
    return True

def neural_learning():
    r=np.random.default_rng(44);net=Network(42);x=r.normal(size=(64,12));actions=r.integers(36,size=64)
    y=np.sin(x[:,0]);before=np.mean((net.predict(x)[np.arange(64),actions]-y)**2)
    for _ in range(300):net.learn(x,actions,y)
    after=np.mean((net.predict(x)[np.arange(64),actions]-y)**2)
    assert after<before*.1
    return dict(before=float(before),after=float(after))

def ack_accounting():
    c=replace(Config(),slots=700,train_n=25,cal_n=80,scenario='interference')
    s=ScaleModel().fit(c,199);cal=calibrate(c,s,200);env=Channel(c,201)
    p=Policy(c,'A-GRCS-ACK',s,cal);errors=0
    for _ in range(c.slots):
        obs,true,ref=env.sample();a=p.schedule(obs);m=p.feedback(obs,true,ref,a)
        errors+=m['failure']
    bound=c.alpha+len(c.degrees)*(c.alpha+c.gamma*(1-c.alpha))/(c.gamma*c.slots)
    identity=c.alpha*c.slots+(len(c.degrees)*c.alpha-sum(x.a for x in p.controllers.values()))/c.gamma
    assert abs(errors-identity)<1e-8 and errors/c.slots<=bound
    return dict(failure_rate=errors/c.slots,bound=bound,identity_gap=abs(errors-identity))

if __name__=='__main__':
    result=dict(exhaustive_max_gap=exhaustive(),adversarial=arbitrary_sequence(),
                causal_interface=interfaces_and_feedback(),neural_training=neural_learning(),ack_accounting=ack_accounting())
    out=Path(__file__).resolve().parents[1]/'docs'/'verification.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

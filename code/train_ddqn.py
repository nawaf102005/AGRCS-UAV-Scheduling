import argparse,json,time
from pathlib import Path
from dataclasses import replace
import numpy as np
from agrcs.model import Config,Channel,ScaleModel
from agrcs.calibration import calibrate
from agrcs.scheduler import Policy
from agrcs.rl import Network,ACTIONS,reward

def validate(net,scale,cal,c):
    scores=[]
    for j,scene in enumerate(['stationary','abrupt','gradual']):
        cc=replace(c,scenario=scene,slots=500)
        env=Channel(cc,800000+j);p=Policy(cc,'DDQN-Match',scale,cal,net)
        rs=[]
        for t in range(cc.slots):
            obs,true,ref=env.sample();a=p.schedule(obs);m=p.feedback(obs,true,ref,a)
            rs.append(reward(m,cc))
        scores.append(np.mean(rs))
    return float(np.mean(scores))

def train(seed,steps,out):
    c=Config();r=np.random.default_rng(900000+seed)
    scale=ScaleModel().fit(c,500000+seed);cal=calibrate(c,scale,510000+seed)
    net=Network(seed);target=Network(seed)
    size=12000;X=np.zeros((size,12));NX=X.copy();A=np.zeros(size,int)
    Y=np.zeros(size);DONE=np.zeros(size);logs=[];best=-np.inf
    start=time.time();loss=0.;eps_return=0.
    for t in range(steps):
        if t%400==0:
            scene=['stationary','abrupt','gradual','recurrent'][int(r.integers(4))]
            cc=replace(c,scenario=scene,slots=400,arrival=float(r.choice([.18,.24,.36])))
            env=Channel(cc,600000+seed*10000+t//400);p=Policy(cc,'DDQN-Match',scale,cal,net)
            obs,true,ref=env.sample();eps_return=0.
        x=p.features(obs,scale.predict(obs))
        epsilon=max(.05,1-.95*t/(steps*.7))
        action=int(r.integers(len(ACTIONS))) if r.random()<epsilon else int(net.predict(x[None])[0].argmax())
        act=p.schedule(obs,action_override=action);metrics=p.feedback(obs,true,ref,act)
        rew=reward(metrics,cc);nobs,ntrue,nref=env.sample()
        nx=p.features(nobs,scale.predict(nobs));j=t%size
        X[j]=x;NX[j]=nx;A[j]=action;Y[j]=rew;DONE[j]=int(t%400==399)
        eps_return+=rew
        if t>=512 and t%4==0:
            ix=r.integers(min(t+1,size),size=64)
            choices=net.predict(NX[ix]).argmax(1)
            td=Y[ix]+.95*(1-DONE[ix])*target.predict(NX[ix])[np.arange(len(ix)),choices]
            loss=net.learn(X[ix],A[ix],td)
        if t%250==0:target.p={k:v.copy() for k,v in net.p.items()}
        if t%400==399:logs.append(dict(step=t+1,return_mean=eps_return/400,loss=loss,epsilon=epsilon))
        if (t+1)%5000==0 or t==steps-1:
            val=validate(net,scale,cal,c)
            net.save(out/f'ddqn_seed{seed}_step{t+1}.npz')
            if val>best:best=val;net.save(out/f'ddqn_seed{seed}.npz')
            print('DDQN',seed,'step',t+1,'validation',round(val,4),flush=True)
            logs.append(dict(step=t+1,validation_reward=val))
        obs,true,ref=nobs,ntrue,nref
    (out/f'training_seed{seed}.json').write_text(json.dumps(dict(seed=seed,steps=steps,
        seconds=time.time()-start,best_validation=best,scale_beta=scale.beta.tolist(),
        config=c.dict(),logs=logs),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--seeds',type=int,default=3);p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'models')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    for seed in range(a.seeds):train(seed,a.steps,a.out)

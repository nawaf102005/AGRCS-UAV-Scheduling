"""Frequency-selective assignment, optimized powers, selectable calibrated graphs."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .model import graph,rate,score,propulsion
from .calibration import Controller

METHODS=['A-GRCS','Frozen-GRCS','Rolling-GRCS','ACI-d2','ACI-Full',
         'Nominal','PF-Power','DDQN-Match','Oracle']

def assign(obs,scale,queue,previous,average,c,degree,threshold,eta=None,pf=False,oracle=None):
    g=graph(obs,degree)
    lower=obs.estimate-threshold*scale if oracle is None else oracle
    # The -infinity quantile is a formal all-miscoverage limit. Cap payloads by queue.
    switch=(previous[:,None]>=0)&(previous[:,None]!=np.arange(c.uavs)[None,:])
    duration=c.slot*(1-c.reference_fraction*degree)*(1-.1*switch)
    powers=np.array(c.powers)
    b=np.minimum(queue[:,None,None,None],duration[:,:,None,None]*rate(lower[:,:,:,None],powers,c))
    w=1+np.minimum(queue/2,20)
    if pf:w=1/np.maximum(average,.1)
    eta=c.eta if eta is None else eta
    util=w[:,None,None,None]*b-eta*c.slot*powers-c.handoff_cost*switch[:,:,None,None]
    best=util.argmax(-1)
    u=np.take_along_axis(util,best[...,None],-1)[...,0]
    payload=np.take_along_axis(b,best[...,None],-1)[...,0]
    edge=np.where(g,u,-1e12).reshape(c.users,-1)
    padded=np.column_stack([edge,np.zeros((c.users,c.users))])
    rows,cols=linear_sum_assignment(padded,maximize=True)
    real=(cols<c.uavs*c.rbs)&(padded[rows,cols]>0)
    k=rows[real];j=cols[real];v=j//c.rbs;r=j%c.rbs
    return dict(k=k,u=v,r=r,p=powers[best[k,v,r]],b=payload[k,v,r],
                duration=duration[k,v],switched=switch[k,v],utility=float(u[k,v,r].sum()),
                degree=degree,threshold=threshold)

class Policy:
    def __init__(self,c,method,scale,cal,network=None):
        self.c=c;self.method=method;self.scale=scale;self.network=network
        self.queue=np.full(c.users,c.initial_queue);self.previous=np.full(c.users,-1,int)
        self.average=np.full(c.users,.1);self.last_failure=0.;self.last_score=0.
        self.energy_backlog=0.
        self.degrees=c.degrees if method in ['A-GRCS','A-GRCS-Budget','A-GRCS-ACK'] else ((c.uavs,) if method=='ACI-Full' else (2,))
        mode='frozen' if method=='Frozen-GRCS' else ('rolling' if method=='Rolling-GRCS' else 'adaptive')
        self.controllers={d:Controller(cal[d],c,mode) for d in self.degrees}
    def features(self,obs,s):
        return np.array([self.queue.mean()/10,np.quantile(self.queue,.9)/20,
                         obs.estimate.mean()/30,obs.estimate.std()/20,s.mean()/10,
                         self.average.mean()/5,self.last_failure,self.last_score/10,
                         self.c.users/100,self.c.arrival,obs.speed.mean()/20,1.],float)
    def schedule(self,obs,oracle=None,action_override=None):
        c=self.c;m=self.method;s=self.scale.predict(obs)
        if m=='DDQN-Match':
            from .rl import ACTIONS
            idx=int(np.argmax(self.network.predict(self.features(obs,s)[None])[0])) if action_override is None else action_override
            d,q,eta=ACTIONS[idx]
            a=assign(obs,s,self.queue,self.previous,self.average,c,d,q,eta)
            a['rl_action']=idx
        elif m in ['Nominal','Oracle']:
            a=assign(obs,s,self.queue,self.previous,self.average,c,2,0,oracle=oracle)
        else:
            price=c.eta+self.energy_backlog if m=='A-GRCS-Budget' else c.eta
            aa=[assign(obs,s,self.queue,self.previous,self.average,c,d,self.controllers[d].threshold(),eta=price,pf=m=='PF-Power') for d in self.degrees]
            a=max(aa,key=lambda a:a['utility'])
        a['scale']=s
        return a
    def feedback(self,obs,true,reference,a):
        c=self.c;k=a['k'];u=a['u'];r=a['r'];p=a['p'];b=a['b']
        cap=a['duration']*rate(true[k,u,r],p,c)
        success=b<=cap+1e-10
        received=np.zeros(c.users);received[k]=b*success
        weighted=float(((1+np.minimum(self.queue/2,20))[k]*b*success).sum())
        self.queue=np.maximum(0,self.queue-received)+obs.arrivals
        self.previous[k]=u;self.average=.95*self.average+.05*received/c.slot
        s=score(obs,reference,a['scale'],a['degree'],c.reference_error)
        e=int(s>a['threshold']);self.last_score=s;self.last_failure=float(np.any(~success))
        if self.method not in ['Nominal','Oracle','DDQN-Match']:
            override=self.last_failure if self.method=='A-GRCS-ACK' else None
            self.controllers[a['degree']].update(s,a['threshold'],error_override=override)
        rf=float(p.sum()*c.slot)
        if self.method=='A-GRCS-Budget':
            self.energy_backlog=max(0.,self.energy_backlog+c.budget_step*(rf-c.rf_budget))
        assert len(set(k))==len(k) and len(set(zip(u,r)))==len(u)
        assert np.all(np.bincount(u,weights=p,minlength=c.uavs)<=c.rbs*max(c.powers)+1e-9)
        # Graph-to-delivery inclusion requires a valid reference envelope.
        if self.method!='Oracle' and np.all(np.abs(reference-true)<=c.reference_error+1e-10):
            assert not np.any(~success) or e==1
        return dict(goodput=received.sum()/c.slot,rf=rf,
                    mission=rf+propulsion(obs.speed).sum()*c.slot,
                    failure=self.last_failure,miscoverage=e,queue=self.queue.mean(),
                    max_queue=self.queue.max(),degree=a['degree'],margin=a['threshold'],
                    idle=float(len(k)==0),links=len(k),failed_links=int((~success).sum()),
                    served=float(success.sum()/c.users),handoffs=int(a['switched'].sum()),
                    utility=weighted-c.eta*rf-c.handoff_cost*a['switched'].sum(),
                    alpha_state=sum(x.a for x in self.controllers.values()),
                    reference_labels=c.users*a['degree']*c.rbs,
                    energy_price=c.eta+self.energy_backlog,
                    delivered=received)

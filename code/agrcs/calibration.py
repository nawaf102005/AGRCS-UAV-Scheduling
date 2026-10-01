"""Separate frozen split calibration and active-graph ACI controllers."""
import math
import numpy as np
from .model import Channel,score

def quantile(values,alpha):
    if alpha<=0:return float('inf')
    if alpha>=1:return -float('inf')
    n=len(values);k=math.ceil((n+1)*(1-alpha))
    return float('inf') if k>n else float(np.partition(values,k-1)[k-1])

def calibrate(c,scale,seed):
    env=Channel(c,seed)
    values={d:[] for d in sorted(set(c.degrees+(c.uavs,)))}
    for _ in range(c.cal_n):
        obs,_,reference=env.sample(independent=True);s=scale.predict(obs)
        for d in values:values[d].append(score(obs,reference,s,d,c.reference_error))
    return {d:np.asarray(v) for d,v in values.items()}

class Controller:
    def __init__(self,values,c,mode='adaptive'):
        self.c=c;self.mode=mode;self.a=c.alpha
        self.buffer=list(values[-c.window:]);self.frozen=quantile(values,c.alpha)
        self.n=0;self.errors=0
    def threshold(self):
        if self.mode=='frozen':return self.frozen
        return quantile(self.buffer,self.a if self.mode=='adaptive' else self.c.alpha)
    def update(self,s,q,error_override=None):
        e=int(s>q) if error_override is None else int(error_override)
        if self.mode=='adaptive':self.a+=self.c.gamma*(self.c.alpha-e)
        self.buffer.append(s)
        if len(self.buffer)>self.c.window:self.buffer.pop(0)
        self.n+=1;self.errors+=e
        return e

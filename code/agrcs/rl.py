"""NumPy Double-DQN: experience replay, target network, Huber loss, Adam.

The macro-action selects candidate degree, standardized margin, and RF price;
the same exact assignment layer enforces resource constraints. This is an
independent DDQN-Match baseline, NOT a reproduction of Yang et al.'s MAD3QN.
"""
import numpy as np
ACTIONS=[(d,q,e) for d in (1,2,4) for q in (0.,1.,2.,3.,4.,6.) for e in (.1,1.)]

class Network:
    def __init__(self,seed=0):
        r=np.random.default_rng(seed);dims=[12,48,48,len(ACTIONS)]
        self.p={}
        for i,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
            self.p[f'w{i}']=r.normal(0,np.sqrt(2/a),(a,b))
            self.p[f'b{i}']=np.zeros(b)
        self.m={k:np.zeros_like(v) for k,v in self.p.items()}
        self.v={k:np.zeros_like(v) for k,v in self.p.items()};self.t=0
    def forward(self,x):
        h=[np.clip(x,-10,10)]
        for i in range(3):
            z=h[-1]@self.p[f'w{i}']+self.p[f'b{i}']
            h.append(np.maximum(z,0) if i<2 else z)
        return h
    def predict(self,x):return self.forward(x)[-1]
    def learn(self,x,action,target):
        h=self.forward(x);err=h[-1][np.arange(len(x)),action]-target
        grad=np.zeros_like(h[-1]);grad[np.arange(len(x)),action]=np.clip(err,-1,1)/len(x)
        gs={}
        for i in (2,1,0):
            gs[f'w{i}']=h[i].T@grad;gs[f'b{i}']=grad.sum(0)
            grad=grad@self.p[f'w{i}'].T
            if i>0:grad*=h[i]>0
        norm=np.sqrt(sum((v*v).sum() for v in gs.values()))
        self.t+=1
        for k in self.p:
            g=gs[k]*min(1,10/max(norm,1e-10))
            self.m[k]=.9*self.m[k]+.1*g;self.v[k]=.999*self.v[k]+.001*g*g
            self.p[k]-=3e-4*(self.m[k]/(1-.9**self.t))/(np.sqrt(self.v[k]/(1-.999**self.t))+1e-8)
        return float(np.where(abs(err)<=1,.5*err**2,abs(err)-.5).mean())
    def save(self,path):np.savez_compressed(path,**self.p)
    @classmethod
    def load(cls,path):
        n=cls();n.p={k:v for k,v in np.load(path).items()};return n

def reward(metrics,c):
    # Delivered payload, RF use, and joint transmission failures; fixed before testing.
    return float(metrics['goodput']*c.slot/(c.users*c.arrival)-.15*metrics['rf']
                 -2*metrics['failure']-.002*metrics['queue'])

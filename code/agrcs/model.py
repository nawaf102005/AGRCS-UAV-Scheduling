"""Temporal, frequency-selective synthetic channel with noisy effective-CSI forecasts.

The decision interface contains no true CSI or simulator error scale. Reference
feedback is revealed only after scheduling, with a declared bounded error.
All resources are orthogonal across UAVs: interference is external, not endogenous.
"""
from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class Config:
    users: int = 24
    uavs: int = 6
    rbs: int = 2
    degrees: tuple = (1, 2, 4)
    slot: float = .1
    total_bw: float = 12e6
    area: float = 1500.
    height: float = 100.
    powers: tuple = (.05, .1, .2, .5)
    alpha: float = .1
    gamma: float = .02
    window: int = 256
    arrival: float = .24  # Mbit/user/slot
    eta: float = .5      # Mbit/J
    initial_queue: float = 2.
    rf_budget: float = .35  # J/slot, soft average budget in optional price controller
    budget_step: float = .5
    handoff_cost: float = .02
    reference_error: float = .5  # deterministic dB bound on reference feedback
    reference_fraction: float = .025  # overhead per candidate UAV, charged to payload
    temporal: float = .85
    error_temporal: float = .7
    scenario: str = 'stationary'
    slots: int = 3000
    train_n: int = 500
    cal_n: int = 1000
    speed: float = 15.
    scale_mode: str = 'learned'

    @property
    def bw(self): return self.total_bw/(self.uavs*self.rbs)
    def dict(self): return asdict(self)

def propulsion(speed):
    """Zeng-Xu-Zhang steady level-flight model; no acceleration or wind correction."""
    v=np.asarray(speed)
    return (79.86*(1+3*v*v/120**2)
            +88.63*np.sqrt(np.sqrt(1+v**4/(4*4.03**4))-v*v/(2*4.03**2))
            +.5*.6*1.225*.05*.503*v**3)

@dataclass
class Observation:
    estimate: np.ndarray  # effective channel dB per W, shape K,U,R
    distance: np.ndarray
    los: np.ndarray
    arrivals: np.ndarray
    speed: np.ndarray

def graph(obs,degree):
    idx=np.argsort(obs.distance,axis=1,kind='stable')[:,:degree]
    g=np.zeros(obs.distance.shape,bool)
    np.put_along_axis(g,idx,True,axis=1)
    return np.broadcast_to(g[:,:,None],obs.estimate.shape)

def rate(ell,power,c):
    # stable softplus computes log2(1+10^(ell/10)*p), including infinite limits
    a=ell*np.log(10)/10+np.log(power)
    return c.bw/1e6*np.logaddexp(0,a)/np.log(2)

class Channel:
    def __init__(self,c,seed):
        self.c=c;self.rng=np.random.default_rng(seed);self.t=0
        self.reset()
    def reset(self):
        c=self.c;r=self.rng
        self.xy=r.uniform(0,c.area,(c.users,2))
        self.uv=r.uniform(.1*c.area,.9*c.area,(c.uavs,2))
        self.dir=r.uniform(0,2*np.pi,c.uavs)
        self.udir=r.uniform(0,2*np.pi,c.users)
        self.shadow=r.normal(size=(c.users,c.uavs))
        self.fade=(r.normal(size=(c.users,c.uavs,4))+1j*r.normal(size=(c.users,c.uavs,4)))/np.sqrt(2)
        self.error=r.normal(size=(c.users,c.uavs,c.rbs))
        self.common=r.normal()
        self.interference=r.normal(size=(c.uavs,c.rbs))
    def sample(self,independent=False):
        c=self.c;r=self.rng
        if independent:self.reset()
        speed=c.speed*(.65+.35*np.sin(self.t*c.slot/12+np.arange(c.uavs))**2)
        self.uv+=c.slot*speed[:,None]*np.column_stack((np.cos(self.dir),np.sin(self.dir)))
        self.xy+=c.slot*1.5*np.column_stack((np.cos(self.udir),np.sin(self.udir)))
        # Reflect at boundaries (turn dynamics not modeled by the propulsion formula).
        for pos,heading in [(self.uv,self.dir),(self.xy,self.udir)]:
            hit=(pos[:,0]<0)|(pos[:,0]>c.area);heading[hit]=np.pi-heading[hit]
            hit=(pos[:,1]<0)|(pos[:,1]>c.area);heading[hit]=-heading[hit]
            pos[:]=c.area-np.abs((pos % (2*c.area))-c.area)
        distance=np.sqrt(((self.xy[:,None]-self.uv[None])**2).sum(-1)+c.height**2)
        theta=np.degrees(np.arcsin(c.height/distance))
        los=1/(1+9.61*np.exp(-.16*(theta-9.61)))
        mean=-20*np.log10(4*np.pi*2e9*distance/299792458)-los-20*(1-los)
        rho=c.temporal
        self.shadow=rho*self.shadow+np.sqrt(1-rho*rho)*r.normal(size=self.shadow.shape)
        self.fade=rho*self.fade+np.sqrt(1-rho*rho)*(r.normal(size=self.fade.shape)+1j*r.normal(size=self.fade.shape))/np.sqrt(2)
        taps=self.fade*np.sqrt(np.array([.55,.25,.15,.05]))
        response=np.einsum('kul,rl->kur',taps,np.exp(-2j*np.pi*np.arange(c.rbs)[:,None]*np.arange(4)[None,:]/max(4,c.rbs)))
        h=(np.sqrt(3)+response)/2  # Rician factor 3 (linear), normalized mean power
        gain=mean[:,:,None]+3*self.shadow[:,:,None]+10*np.log10(np.maximum(abs(h)**2,1e-12))
        old_interference=self.interference.copy()
        self.interference=.9*self.interference+np.sqrt(1-.9**2)*r.normal(size=self.interference.shape)
        noise_db=-174+10*np.log10(c.bw)+7-30
        log_i=3*self.interference
        estimate_i=3*old_interference+r.normal(0,.8,size=old_interference.shape)
        progress=self.t/max(c.slots,1)
        shift=1.
        if not independent:
            if c.scenario=='abrupt' and progress>=.35:shift=1.8
            if c.scenario=='gradual':shift=1+.8*min(1,progress/.7)
            if c.scenario=='recurrent':shift=1.8 if int(progress*6)%2 else 1.
            if c.scenario=='interference' and progress>=.35:log_i=log_i+10
        a=c.error_temporal
        self.error=a*self.error+np.sqrt(1-a*a)*r.normal(size=self.error.shape)
        self.common=a*self.common+np.sqrt(1-a*a)*r.normal()
        # Neither this simulator scale nor shift indicator is returned to the scheduler.
        sigma=2+3*(1-los)
        estimation_error=shift*sigma[:,:,None]*(np.sqrt(.35)*self.common+np.sqrt(.65)*self.error)
        ell=gain-noise_db-10*np.log10(1+10**(log_i[None]/10))
        estimate=gain+estimation_error-noise_db-10*np.log10(1+10**(estimate_i[None]/10))
        reference=ell+r.uniform(-c.reference_error,c.reference_error,size=ell.shape)
        arrivals=c.arrival*r.uniform(.5,1.5,c.users)
        self.t+=1
        return Observation(estimate,distance,los,arrivals,speed),ell,reference

class ScaleModel:
    """Ridge log-residual regression, fitted exclusively on independent development data."""
    @staticmethod
    def features(obs):
        d=np.broadcast_to(np.log(obs.distance/500)[:,:,None],obs.estimate.shape)
        l=np.broadcast_to((1-obs.los)[:,:,None],obs.estimate.shape)
        x=obs.estimate/30
        return np.stack([np.ones_like(x),d,l,x,x*x],axis=-1)
    def fit(self,c,seed):
        self.unit=c.scale_mode=='unit'
        if self.unit:
            self.beta=np.zeros(5)
            return self
        env=Channel(c,seed);xs=[];ys=[]
        for _ in range(c.train_n):
            obs,_,ref=env.sample(independent=True)
            xs.append(self.features(obs).reshape(-1,5))
            ys.append(np.log(np.maximum(abs(obs.estimate-ref),.05)).ravel())
        X=np.concatenate(xs);y=np.concatenate(ys)
        self.beta=np.linalg.solve(X.T@X+1e-2*np.eye(5),X.T@y)
        return self
    def predict(self,obs):
        if getattr(self,'unit',False):return np.ones_like(obs.estimate)
        # Gaussian absolute-log correction is an efficiency heuristic, never a validity assumption.
        return np.clip(np.exp(self.features(obs)@self.beta+.63518),.5,30.)

def score(obs,reference,scale,degree,eps):
    # ref-eps <= truth: this conservative reference envelope certifies actual capacity.
    return float(((obs.estimate-reference+eps)/scale)[graph(obs,degree)].max())

"""Explicit, uncalibrated queue model for three signal locations on 9th Street."""
from dataclasses import dataclass
import numpy as np

DT=10
STEPS=360
NAMES=['9th & Q','9th & P','9th & O']
SCENARIOS={
 'standard':dict(label='Post-game pulse',demand=1.0,shuttle=0.0,capacity=1.0),
 'surge':dict(label='25% higher event arrival rate',demand=1.25,shuttle=0.0,capacity=1.0),
 'shuttle':dict(label='25% lower event arrival rate (shuttle assumption)',demand=1.0,shuttle=.25,capacity=1.0),
 'restriction':dict(label='P Street discharge reduced by half',demand=1.0,shuttle=0.0,capacity=.5),
}

def make_arrivals(seed, demand=1., shuttle=0.):
    rng=np.random.default_rng(seed)
    a=np.zeros((STEPS,3,2),dtype=np.int64)
    for t in range(180):
        # Synthetic 30-minute post-game pulse plus modest cross traffic.
        event=5.5*np.exp(-((t-45)/43)**2)*(1-shuttle)*demand
        rates=np.array([[.8+event,1.3],[.3,1.6],[.3,1.5]])*rng.uniform(.9,1.1,(3,2))
        a[t]=rng.poisson(rates)
    return a

@dataclass
class State:
    q: np.ndarray
    phase: np.ndarray
    age: np.ndarray
    arrivals: int=0
    departed: int=0
    queue_seconds: float=0.
    switches: int=0

def initial_state():
    return State(np.zeros((3,2),dtype=float),np.zeros(3,dtype=int),np.full(3,2,dtype=int))

def capacity(factor=1.):
    c=np.full((3,2),8.,dtype=float)
    c[1,0]*=factor
    return c

def observe(state, cap):
    # Six queues, three phases, three green ages, three corridor capacities.
    return np.concatenate([np.minimum(state.q.reshape(-1)/60,4),state.phase,
                           np.minimum(state.age/8,1),cap[:,0]/8]).astype(np.float32)

def teacher(state,cap):
    downstream=np.array([state.q[1,0],state.q[2,0],0.])
    scores=np.stack([np.maximum(state.q[:,0]-.7*downstream,0)*cap[:,0]/8,state.q[:,1]],axis=1)
    scores[np.arange(3),state.phase]+=4 # hysteresis for switch loss
    return scores.argmax(1)

def fixed(t):
    # Same illustrative 80-second cycle at all nodes: 50s corridor, 30s cross.
    return np.full(3,0 if t%8<5 else 1,dtype=int)

def step(state,action,arrivals,cap):
    action=np.asarray(action,dtype=int).copy()
    if action.shape!=(3,) or not np.isin(action,[0,1]).all():
        raise ValueError('Three binary phase requests required')
    state.q+=arrivals;state.arrivals+=int(np.sum(arrivals))
    # Generic guard shared by every controller: minimum 20s, maximum 80s if
    # conflicting traffic waits. Switching costs a whole 10s coarse time step.
    action=np.where(state.age<2,state.phase,action)
    other=1-state.phase
    must=(state.age>=8)&(state.q[np.arange(3),other]>0)
    action=np.where(must,other,action)
    changed=action!=state.phase
    state.switches+=int(changed.sum());state.phase=action
    state.age=np.where(changed,0,state.age+1)
    flow=np.zeros((3,2))
    for i in range(3):
        if not changed[i]:
            j=state.phase[i]
            flow[i,j]=min(state.q[i,j],cap[i,j])
            if j==0 and i<2:
                flow[i,j]=min(flow[i,j],max(0,60-state.q[i+1,0]))
    # Simultaneous discharge; downstream receives transfers for the next step.
    state.q-=flow
    state.q[1:,0]+=flow[:-1,0]
    state.departed+=int(flow[:,1].sum()+flow[-1,0])
    state.queue_seconds+=float(state.q.sum())*DT
    assert np.all(state.q>=0)
    assert abs(state.q.sum()+state.departed-state.arrivals)<1e-8
    return flow,changed

def simulate(arrivals,cap,policy,record=False):
    s=initial_state();trace=[];maxq=0
    for t,a in enumerate(arrivals):
        action,activity=policy(observe(s,cap),s,cap,t)
        flow,changed=step(s,action,a,cap)
        maxq=max(maxq,float(s.q.sum()))
        if record:
            trace.append(dict(t=(t+1)*DT,q=s.q.tolist(),phase=s.phase.tolist(),clearance=changed.tolist(),
                              flow=flow.tolist(),departed=s.departed,arrivals=s.arrivals,
                              queue_seconds=s.queue_seconds,activity=activity))
    return dict(queue_seconds_per_arrival=s.queue_seconds/max(1,s.arrivals),
                total_queue_hours=s.queue_seconds/3600,departed=s.departed,arrivals=s.arrivals,
                residual_queue=float(s.q.sum()),peak_queue=maxq,switches=s.switches),trace

def sample_training(seed,episodes):
    rng=np.random.default_rng(seed);xs=[];ys=[]
    for ep in range(episodes):
        demand=rng.uniform(.65,1.35);shuttle=rng.uniform(0,.3)
        cap=capacity(rng.choice([.5,.75,1.]))
        arrivals=make_arrivals(int(rng.integers(1,1000000000)),demand,shuttle)
        s=initial_state()
        # Mixture of controllers exposes training to off-teacher states.
        mode=ep%3
        for t,a in enumerate(arrivals[:240]):
            obs=observe(s,cap);target=teacher(s,cap)
            xs.append(obs);ys.append(target)
            action=target if mode==0 else fixed(t) if mode==1 else rng.integers(0,2,3)
            step(s,action,a,cap)
    return np.asarray(xs,dtype=np.float32),np.asarray(ys,dtype=np.float32)

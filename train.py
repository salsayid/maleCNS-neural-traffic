"""Imitation learning of a queue heuristic using measured/rewired fly circuits."""
from pathlib import Path
import copy,json,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
from connectome import load_circuit,rewire
from traffic import sample_training,make_arrivals,capacity,simulate,teacher,fixed,SCENARIOS

class Policy(nn.Module):
    def __init__(self,n,e):
        super().__init__();N=len(n)
        self.register_buffer('src',torch.tensor(e.source.to_numpy(),dtype=torch.long))
        self.register_buffer('dst',torch.tensor(e.target.to_numpy(),dtype=torch.long))
        counts=torch.tensor(e.weight.to_numpy(),dtype=torch.float32)
        total=torch.zeros(N).index_add_(0,self.dst,counts).clamp_min(1)
        self.register_buffer('base',1.5*counts/total[self.dst]*torch.tensor(n.sign.to_numpy(),dtype=torch.float32)[self.src])
        self.register_buffer('type_ids',torch.tensor(n.type_index.to_numpy(),dtype=torch.long))
        self.register_buffer('inputs',torch.tensor(np.flatnonzero(n.type.isin(['L1','L3']).to_numpy()),dtype=torch.long))
        self.register_buffer('outputs',torch.tensor(np.flatnonzero(n.type.str.startswith('T4').to_numpy()),dtype=torch.long))
        self.encoder=nn.Linear(15,len(self.inputs))
        self.edge_scale=nn.Parameter(torch.zeros(len(e)))
        self.bias=nn.Parameter(torch.zeros(10))
        self.leak=nn.Parameter(torch.linspace(-1.5,.5,10))
        self.decoder=nn.Linear(len(self.outputs),3)

    def forward(self,x,return_state=False,ablate=None):
        N=len(self.type_ids)
        w=x.new_zeros(N,N).index_put((self.dst,self.src),self.base*self.edge_scale.clamp(-2,2).exp())
        drive=x.new_zeros(len(x),N).index_copy(1,self.inputs,self.encoder(x))
        h=x.new_zeros(len(x),N)
        alpha=.05+.9*self.leak[self.type_ids].sigmoid()
        bias=self.bias[self.type_ids]
        alive=x.new_ones(N)
        if ablate is not None:alive[ablate]=0
        for _ in range(8):
            h=((1-alpha)*h+alpha*torch.tanh(h@w.T+drive+bias))*alive
        logits=self.decoder(h[:,self.outputs])
        return (logits,h) if return_state else logits

class MLP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(15,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,3))
    def forward(self,x):return self.net(x)

@torch.no_grad()
def score(model,x,y):
    logits=torch.cat([model(batch) for batch in x.split(512)])
    return float(F.binary_cross_entropy_with_logits(logits,y)),float(((logits>0)==(y>0)).float().mean())

def train(model,seed,train_data,val_data,epochs=30):
    optim=torch.optim.Adam(model.parameters(),lr=.003)
    gen=torch.Generator().manual_seed(seed)
    x,y=train_data;vx,vy=val_data
    best_loss=float('inf');best=None;history=[];best_epoch=0
    for epoch in range(1,epochs+1):
        total=0
        for ids in torch.randperm(len(x),generator=gen).split(256):
            loss=F.binary_cross_entropy_with_logits(model(x[ids]),y[ids])
            optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optim.step()
            total+=float(loss.detach())*len(ids)
        vl,acc=score(model,vx,vy)
        history.append(dict(epoch=epoch,train_loss=total/len(x),validation_loss=vl,validation_agreement=acc))
        if vl<best_loss:best_loss=vl;best=copy.deepcopy(model.state_dict());best_epoch=epoch
        if epoch==1 or epoch%10==0:print(f' epoch {epoch}: loss {total/len(x):.4f}; validation agreement {acc:.3%}',flush=True)
    model.load_state_dict(best);return history,best_epoch

def callback(model,n,ablate=None,record=False):
    @torch.no_grad()
    def act(obs,state,cap,t):
        x=torch.tensor(obs[None,:])
        if isinstance(model,Policy):
            logits,h=model(x,return_state=True,ablate=ablate)
            activity=[round(float(h[0,model.type_ids==i].mean()),5) for i in range(10)] if record else []
        else:logits=model(x);activity=[]
        return (logits[0]>0).numpy().astype(int),activity
    return act

def main():
    torch.set_num_threads(2);out=ROOT/'results';out.mkdir(exist_ok=True)
    n,e=load_circuit()
    tx,ty=sample_training(4101,90);vx,vy=sample_training(4102,18)
    np.savez_compressed(ROOT/'data/training.npz',x=tx,y=ty,val_x=vx,val_y=vy)
    train_data=(torch.tensor(tx),torch.tensor(ty));val_data=(torch.tensor(vx),torch.tensor(vy))
    metrics=dict(method='Supervised imitation of downstream-aware queue heuristic; not reinforcement learning',
        train_examples=len(tx),validation_examples=len(vx),epochs=30,seeds=[11,22,33],
        train_data_seed=4101,validation_data_seed=4102,evaluation_seeds=list(range(5100,5120)),runs=[],baselines={})
    all_models={};start=time.monotonic()
    for seed in metrics['seeds']:
        for kind in ['connectome','rewired','mlp']:
            torch.manual_seed(seed)
            model=MLP() if kind=='mlp' else Policy(n,e if kind=='connectome' else rewire(e,n,seed)[0])
            print(kind,seed,flush=True)
            history,best_epoch=train(model,seed,train_data,val_data)
            torch.save(model.state_dict(),out/f'{kind}_{seed}.pt')
            run=dict(kind=kind,seed=seed,history=history,best_epoch=best_epoch,parameters=sum(p.numel() for p in model.parameters()),scenarios={})
            for scenario,config in SCENARIOS.items():
                runs=[]
                for evseed in metrics['evaluation_seeds']:
                    arrivals=make_arrivals(evseed,config['demand'],config['shuttle'])
                    result,_=simulate(arrivals,capacity(config['capacity']),callback(model,n))
                    runs.append(dict(seed=evseed,**result))
                run['scenarios'][scenario]=runs
            metrics['runs'].append(run)
            all_models[(kind,seed)]=model
            (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    for name,cb in [('fixed',lambda o,s,c,t:(fixed(t),[])),('heuristic',lambda o,s,c,t:(teacher(s,c),[]))]:
        metrics['baselines'][name]={}
        for scenario,config in SCENARIOS.items():
            metrics['baselines'][name][scenario]=[dict(seed=seed,**simulate(make_arrivals(seed,config['demand'],config['shuttle']),capacity(config['capacity']),cb)[0]) for seed in metrics['evaluation_seeds']]
    # Fixed seed 11 for illustrative traces and ablations; no test-based selection.
    traces={};ablation={}
    for scenario,config in SCENARIOS.items():
        arrivals=make_arrivals(5100,config['demand'],config['shuttle']);cap=capacity(config['capacity']);traces[scenario]={}
        for name in ['connectome','rewired','mlp','fixed','heuristic']:
            cb=(lambda o,s,c,t:(fixed(t),[])) if name=='fixed' else (lambda o,s,c,t:(teacher(s,c),[])) if name=='heuristic' else callback(all_models[(name,11)],n,record=True)
            result,trace=simulate(arrivals,cap,cb,record=True)
            traces[scenario][name]=dict(metrics=result,steps=trace)
    for typ in ['Mi1','Mi4','T4a','all_T4']:
        mask=torch.tensor((n.type.str.startswith('T4') if typ=='all_T4' else n.type==typ).to_numpy().copy())
        ablation[typ]=[dict(seed=seed,**simulate(make_arrivals(seed),capacity(),callback(all_models[('connectome',11)],n,ablate=mask))[0]) for seed in metrics['evaluation_seeds']]
    metrics['ablations_seed11']=ablation;metrics['elapsed_seconds']=time.monotonic()-start
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2));(out/'traces.json').write_text(json.dumps(traces,separators=(',',':')))
    # Save a compact numerical reference for exported browser checks.
    model=all_models[('connectome',11)]
    with torch.no_grad():reference=dict(inputs=vx[:16].tolist(),logits=model(torch.tensor(vx[:16])).tolist())
    (out/'reference.json').write_text(json.dumps(reference))
    print('Finished:',metrics['elapsed_seconds'],'seconds',flush=True)

if __name__=='__main__':main()

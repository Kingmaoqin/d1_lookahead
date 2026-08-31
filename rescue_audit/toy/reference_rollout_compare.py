"""Independent slow rollout compared against `src/policy.py` on a fake DLM."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import policy

MASK=policy.MASK_TOKEN_ID
MOD=1<<64
C_ADD=0x9E3779B97F4A7C15; C_M1=0xBF58476D1CE4E5B9
C_M2=0x94D049BB133111EB; C_SEED=0x2545F4914F6CDD1D


def sm(x):
    x=(x+C_ADD)%MOD; x=((x^(x>>30))*C_M1)%MOD
    x=((x^(x>>27))*C_M2)%MOD
    return (x^(x>>31))%MOD


def uni(seed,stream,pos,slot):
    z=(seed*C_SEED+sm(stream+1)+pos*C_M2+slot*C_ADD)%MOD
    return (((sm(z)>>11)+.5)/(1<<53))


def gum(u): return -math.log(-math.log(min(max(u,1e-7),1-1e-7)))


def logits_np(ids):
    ids=np.asarray(ids); L=len(ids); out=np.empty((L,4))
    visible=sum((j+1)*(int(x)+1) for j,x in enumerate(ids) if x!=MASK)
    for i in range(L):
        for v in range(4):
            out[i,v]=.23*(v+1)*(i+1)+.017*visible*(v-1.5)-.11*(v==((i+visible)%4))
    return out


class Fake(torch.nn.Module):
    def forward(self,ids,output_hidden_states=False):
        z=np.stack([logits_np(x.tolist()) for x in ids.cpu()])
        return torch.tensor(z,dtype=torch.float32,device=ids.device)


def reference(ids,mask,seed,horizon):
    ids=list(ids); mask=list(mask); L=len(ids); trace=[]; mc=rb=first_mc=first_rb=0.
    order=[uni(seed,0,p,0) for p in range(L)]
    for step in range(horizon):
        if not any(mask): break
        lg=logits_np(ids); pos=max((p for p in range(L) if mask[p]),key=lambda p:order[p])
        z=lg[pos]-np.log(np.exp(lg[pos]).sum())
        idx=np.argsort(-z)[:4]
        tok=max(idx,key=lambda v:z[v]+gum(uni(seed,1,pos,int(v))))
        p=np.exp(z[idx]); p=p/p.sum(); r=float(np.sum(p*z[idx]))
        val=float(z[tok]); mc+=val; rb+=r
        if step==0: first_mc,first_rb=val,r
        ids[pos]=int(tok); mask[pos]=False; trace.append((pos,int(tok)))
    return {'ids':ids,'trace':trace,'path_ll':mc,'path_ll_rb':rb,
            'first_ll':first_mc,'first_ll_rb':first_rb,'n_commit':len(trace)}


def main():
    init=[0,MASK,MASK,MASK]; m=[False,True,True,True]; seeds=[3,17,101]
    cfg=policy.PiRefConfig(seq_len=4,prefix_len=1,top_k=4,temperature=1.,order='ancestral')
    ids=torch.tensor([init]*len(seeds)); mask=torch.tensor([m]*len(seeds)); sd=torch.tensor(seeds)
    got=policy.rollout(Fake(),ids,mask,sd,cfg,horizon=3,return_trace=True)
    refs=[reference(init,m,s,3) for s in seeds]
    trace=[[(int(got['trace'][t][0][b]),int(got['trace'][t][1][b])) for t in range(3)] for b in range(len(seeds))]
    for b,r in enumerate(refs):
        assert got['ids'][b].tolist()==r['ids']
        assert trace[b]==r['trace']
        for k in ('path_ll','path_ll_rb','first_ll','first_ll_rb'):
            assert abs(float(got[k][b])-r[k])<2e-6,(b,k,float(got[k][b]),r[k])
        assert int(got['n_commit'][b])==r['n_commit']
    out={'status':'VALIDATED','n_seeds':len(seeds),'traces':trace,
         'final_ids':got['ids'].tolist(),'max_numeric_tolerance':2e-6}
    dst=ROOT/'rescue_audit/results/reference_rollout_compare.json'; dst.write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))


if __name__=='__main__': main()

"""Phase R1 exploratory screen for P0-P13 on existing label shards."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'rescue_audit')]
import dataset as D
import probes as P
import probe_suite as R


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tags',nargs='+',default=['a3','b3'])
    ap.add_argument('--target',default='A_pertok')
    ap.add_argument('--layer',type=int,default=None)
    ap.add_argument('--seeds',type=int,default=3)
    ap.add_argument('--pca_dim',type=int,default=64)
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--include_stale_history',action='store_true',
                    help='include legacy C2 flip/persistence columns even though '
                         'the independent audit found them shifted by one state')
    ap.add_argument('--prompt_stratum',choices=['all','natural','informative'],
                    default='all', help='task-utility prompt substrate subset')
    ap.add_argument('--out',default=None)
    args=ap.parse_args()
    d=D.load_labels(args.tags)
    if args.prompt_stratum != 'all':
        if 'prompt_stratum' not in d:
            raise ValueError('--prompt_stratum requires task-utility shards')
        want = 0 if args.prompt_stratum == 'natural' else 1
        keep = d['prompt_stratum'] == want
        n0 = len(keep)
        for key,val in list(d.items()):
            if isinstance(val,np.ndarray) and val.ndim and len(val) == n0:
                d[key] = val[keep]
    y=d[args.target].astype(np.float32)
    ceiling=None
    sk={'A_pertok':'A_full_seeds','A_future':'A_future_seeds','A_task':'A_task_seeds'}.get(args.target)
    if sk in d: ceiling=P.noise_ceiling(d[sk])[0]
    out=Path(args.out or ROOT/'rescue_audit/results'/('screen_'+'_'.join(args.tags)+'_'+args.target))
    out.mkdir(parents=True,exist_ok=True)
    report={'config':vars(args),'n_examples':len(y),'n_docs':len(np.unique(d['doc_id'])),'noise_ceiling':ceiling,'results':[],'unavailable':[]}

    for seed in range(args.seeds):
        sp=D.doc_splits(d,seed); tr,va,te=sp['train'],sp['val'],sp['test']
        if args.include_stale_history:
            xc=D.block(d,'cheap').astype(np.float32)
        else:
            # Existing shards cannot be silently treated as having correctly
            # aligned history.  Exclude the two affected coordinates until a
            # deterministic backfill is completed; save this choice in config.
            xc=np.concatenate([d['C1'],d['C2'][:,:8],d['C2'][:,10:],d['C3']],1).astype(np.float32)
        # P0 chooses layer on validation only unless explicitly fixed.
        if args.layer is None:
            vals=[]
            for l in range(d['n_layers']):
                hi=D.block(d,'H_local',l)
                m=P.fit_linear_2block(xc[tr],hi[tr],y[tr],xc[va],hi[va],y[va])
                vals.append(m['val_r2'])
            layer=int(np.argmax(vals))
        else: layer=args.layer
        hi=D.block(d,'H_local',layer); hg=D.block(d,'H_global',layer)
        hip,hgp,pi,pg=R.pca_pair(hi,hg,tr,args.pca_dim,seed)

        def add(probe,pred,hp,features):
            report['results'].append({'seed':seed,'probe':probe,'features':features,'layer':layer,
                'metrics':R.decision_metrics(y[te],pred,d['state_id'][te],ceiling),'hp':hp})

        # P0 anchors.
        for name,x in [('P0_cheap',xc),('P0_hi',hi),('P0_hg',hg),('P0_cheap_hi',np.c_[xc,hi]),('P0_cheap_hi_hg',np.c_[xc,hi,hg])]:
            m=R.fit_ridge(x[tr],y[tr],x[va],y[va]); add(name,R.predict_sklearn(m,x[te]),{'alpha':m['alpha']},name[3:])
        # P1 cross-fitted residualized linear.
        pred,hp=R.residualized_linear(xc,hi,y,d['doc_id'],tr,va,te); add('P1_residual_hi',pred,hp,'cheap_oof_residual+hi')
        pred,hp=R.residualized_linear(xc,np.c_[hi,hg],y,d['doc_id'],tr,va,te); add('P1_residual_hi_hg',pred,hp,'cheap_oof_residual+[hi,hg]')

        cfg=R.SuiteConfig(seed=seed,pca_dim=args.pca_dim,epochs=25 if args.quick else 100,patience=8 if args.quick else 18)
        # P2 rank selected on validation would require nested fitting; exploratoryly save every rank.
        ranks=(2,4,8) if args.quick else (2,4,8,16,32,64)
        for rank in ranks:
            pred,hp=R.fit_torch_score('bilinear',xc,hip,hgp,y,d['state_id'],tr,va,te,cfg,rank=rank)
            add(f'P2_bilinear_r{rank}',pred,hp,'cheap+PCA(hi)^TUV^TPCA(hg)')
        # Required state-conditioned controls.
        shuffled=hgp.copy(); np.random.default_rng(seed+91).shuffle(shuffled)
        pred,hp=R.fit_torch_score('bilinear',xc,hip,shuffled,y,d['state_id'],tr,va,te,cfg,rank=8); add('P2_shuffled_hg',pred,hp,'shuffled_hg_control')
        gaussian=np.random.default_rng(seed+92).standard_normal(hgp.shape).astype(np.float32)
        pred,hp=R.fit_torch_score('bilinear',xc,hip,gaussian,y,d['state_id'],tr,va,te,cfg,rank=8); add('P2_gaussian_hg',pred,hp,'gaussian_hg_control')

        for name,kind,obj in [('P3_FiLM','film','mse'),('P4_relational_MLP','relational','mse'),('P8_RankNet','relational','pairwise'),('P9_ListNet','relational','listwise')]:
            pred,hp=R.fit_torch_score(kind,xc,hip,hgp,y,d['state_id'],tr,va,te,cfg,objective=obj); add(name,pred,hp,'cheap+relational_PCA_hidden')

        # P5/P6 nonlinear diagnostics on relational products, and capacity-matched cheap controls.
        rel=np.c_[xc,hip,hgp,hip-hgp,np.abs(hip-hgp),hip*hgp].astype(np.float32)
        for kind in (('rbf',) if args.quick else ('rbf','poly2','nystroem')):
            pred,hp=R.kernel_probe(rel,y,tr,va,te,kind=kind,seed=seed); add('P5_'+kind,pred,hp,'relational_PCA_hidden')
        pred,hp=R.boosting_probe(rel,y,tr,va,te,seed); add('P6_boosting',pred,hp,'relational_PCA_hidden')
        pred,hp=R.boosting_probe(xc,y,tr,va,te,seed); add('P6_boosting_cheap',pred,hp,'cheap_capacity_control')

        # P7 direct linear pairwise rankers.
        pred,hp=R.pairwise_logistic(hip,y,d['state_id'],tr,va,te,seed); add('P7_pairwise_hi',pred,hp,'PCA(hi_i-hi_j)')
        pred,hp=R.pairwise_logistic(np.c_[hip,hip*hgp],y,d['state_id'],tr,va,te,seed); add('P7_pairwise_relational',pred,hp,'[hi,hi*hg]')

        # P10 exploratory action identity via fixed hash embedding. Genuine model
        # token/unembedding vectors require a separate backbone extraction pass.
        tok=d['proposed_token'].astype(np.uint64)
        rr=np.arange(32,dtype=np.uint64)[None]
        act=np.sin(((tok[:,None]*(rr*2+1)+17)%104729)/104729*2*np.pi).astype(np.float32)
        pred,hp=R.fit_torch_score('relational',xc,hip,hgp,y,d['state_id'],tr,va,te,cfg,action=act); add('P10_action_hash_diagnostic',pred,hp,'relational+fixed_token_hash')
        report['unavailable'].append({'seed':seed,'probe':'P10_genuine_embedding','reason':'embedding/unembedding vectors were not stored; requires frozen-backbone extraction'})

        # P11 is explicitly unavailable rather than leaking a future state.
        report['unavailable'].append({'seed':seed,'probe':'P11_temporal_delta','reason':'H_{t-1} for the same candidate position was not stored; no t+1 proxy used'})

        # P12 predefined adjacent layer combinations and a learned scalar mix.
        adj=sorted(set([max(0,layer-2),max(0,layer-1),layer,min(d['n_layers']-1,layer+1),min(d['n_layers']-1,layer+2)]))
        for l2 in adj:
            x=np.c_[xc,D.block(d,'H_local',layer),D.block(d,'H_local',l2)]
            m=R.fit_ridge(x[tr],y[tr],x[va],y[va]); add(f'P12_adjacent_{layer}_{l2}',R.predict_sklearn(m,x[te]),{'alpha':m['alpha']},'cheap+adjacent_layers')
        # PCA each layer with the selected-layer PCA: common coordinate system.
        layerproj=np.stack([pi.transform(d['H_i'][:,l].astype(np.float32)) for l in range(d['n_layers'])],1).astype(np.float32)
        pred,hp=R.fit_torch_score('scalar_mix',xc,hip,hgp,y,d['state_id'],tr,va,te,cfg,layers=layerproj); add('P12_scalar_mix',pred,hp,'learned_layer_mix')

        # P13 supervised low-dimensional diagnostic.
        pred,hp=R.pls_probe(rel,y,tr,va,te); add('P13_PLS',pred,hp,'relational_PCA_hidden')

        (out/'partial.json').write_text(json.dumps(report,indent=2,default=float))
    (out/'report.json').write_text(json.dumps(report,indent=2,default=float))
    print(out/'report.json')


if __name__=='__main__': main()

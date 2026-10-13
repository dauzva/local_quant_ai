"""Re-score a run's saved strategies with the CURRENT scoring (beta-adjusted when scoring.beta_adjust)
and print the position-correlation matrix between them. Run from the project root:

    .venv\Scripts\python.exe -I scripts/rescore_run.py [results/<run_id>]

Reads OSS2 of strategies that were already tested there (post-mortem only; do not select on it).
"""
import sys, glob, os; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from src.config import load_config
from src.universe import *
from src.universe import _segment_summary
from src.strategy_sandbox import compile_strategy
cfg=load_config('config.yaml'); U=load_universe(cfg); ev=build_eval_settings(cfg)
print(len(U),'symbols; beta_adjust',ev.beta_adjust)
files=sorted(glob.glob((sys.argv[1] if len(sys.argv) > 1 else 'results/local_quant_run_20261013_072057_09eb88') + '/strategies/*/strat_gen_*.py'))
vecs={}; rows=[]
for f in files:
    c=compile_strategy(open(f).read()); n=os.path.basename(os.path.dirname(f))
    sig={sd.symbol:compute_signal_arrays(c,sd) for sd in U}
    vecs[n]=position_vector(c,U,cfg,sig)
    r={'strategy':n}
    for seg in ('oss1','oss2'):
        o=_segment_summary([segment_symbol(c,sd,ev,cfg,getattr(sd,seg),sigs=sig[sd.symbol]) for sd in U])
        r[seg+'_adj']=o['mean_sharpe']; r[seg+'_raw']=o['raw_mean_sharpe']; r[seg+'_h1']=o.get('mean_sharpe_h1'); r[seg+'_h2']=o.get('mean_sharpe_h2')
    isr=evaluate_universe(c,U,ev,cfg,probe=None,sig_cache=sig)
    r['is_adj']=isr.summary['score']
    rows.append(r)
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(pd.DataFrame(rows).round(3).to_string())
names=list(vecs); M=pd.DataFrame([[position_correlation(vecs[a],vecs[b]) for b in names] for a in names],index=names,columns=[str(i) for i in range(len(names))])
print(M.round(2).to_string())

"""Adversarial verification: does greedy-by-confidence de-overlap inflate reported acc vs random priority at same coverage?"""
import sys, numpy as np
sys.argv = ["x"]  # avoid production __main__ side effects on import
import min1_production as M

def deoverlap(ts, idx_pool, priority, gap=M.GAP):
    """Greedy non-overlap over idx_pool (already the candidate set), ordered by DESCENDING priority."""
    order = idx_pool[np.argsort(-priority)]
    tmin = int(ts[order].min()); span = int(ts[order].max()-tmin)+gap+2
    blk = np.zeros(span, bool); take = []
    for i in order:
        t = int(ts[i]-tmin)
        if blk[t]: continue
        take.append(i); blk[max(0,t-gap+1):t+gap] = True
    return np.sort(np.array(take, dtype=int))

p,L,G,C,S = M._load()
for sp,label in (("test","TEST"),("oos","OOS")):
    b = M.load_split(sp); X,y,mag,valid,ts,idx = M.prep(b)
    pr = M._blend(p,L,G,C,S,X)
    bbw=X["bbw1800"].values; rel=X["rel_ratio"].values; r300=X["ret300"].values
    gate = valid&(bbw<=p["bbw1800_q67"])&(rel>=p["rel_tighten"])&(np.sign(pr-0.5)==-np.sign(r300))
    conf = np.abs(pr-0.5); cand = gate&(conf>=p["conf_thr"])
    pool = np.where(cand)[0]
    correct_all = ((pr>0.5).astype(int)==y)

    # candidate-pool accuracy (all confident bars, pre-nonoverlap)
    pool_acc = correct_all[pool].mean()

    # production greedy (reproduce exactly)
    tr_prod = M.nonoverlap(ts, np.where(cand,conf,-1.0), 0.0); tr_prod = tr_prod[cand[tr_prod]]
    greedy_acc = correct_all[tr_prod].mean(); greedy_n = len(tr_prod)

    # my greedy reimpl (sanity: should match production)
    tr_g = deoverlap(ts, pool, conf[pool], gap=M.GAP)
    g2_acc = correct_all[tr_g].mean(); g2_n = len(tr_g)

    # random priority, 20 seeds
    raccs=[]; rns=[]
    for seed in range(20):
        rng = np.random.default_rng(seed)
        prio = rng.random(len(pool))
        tr_r = deoverlap(ts, pool, prio, gap=M.GAP)
        raccs.append(correct_all[tr_r].mean()); rns.append(len(tr_r))
    raccs=np.array(raccs); rns=np.array(rns)

    print(f"\n=== {label} ===")
    print(f"  pool size (confident bars)      : {len(pool)}   pool_acc={pool_acc:.4f}")
    print(f"  PRODUCTION greedy               : n={greedy_n}  acc={greedy_acc:.4f}")
    print(f"  my-greedy (gap=60)              : n={g2_n}  acc={g2_acc:.4f}")
    print(f"  RANDOM priority (20 seeds)      : n={rns.mean():.1f}+-{rns.std():.1f}  acc={raccs.mean():.4f}+-{raccs.std():.4f}")
    print(f"  GREEDY - RANDOM (pts)           : {100*(greedy_acc-raccs.mean()):+.2f}")
    print(f"  GREEDY - POOL   (pts)           : {100*(greedy_acc-pool_acc):+.2f}")

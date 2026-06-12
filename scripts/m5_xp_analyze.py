"""Offline conditional analysis of the cross-pair model diagnostics (models/m5xp_<mode>_diag.npz).
Question: is there ANY conditioning region (cross-pair agreement / order-flow confirmation / low dispersion / regime)
where 5-min EURUSD direction is STABLE >=0.58-0.65 across test24/test25/oos? Gates are SELECTED ON VAL (two protocols:
VAL-max and transfer-robust = stable across VAL 2022/2023 halves), then reported held-out. No cherry-picking on held-out.
"""
import sys, numpy as np
MODE=sys.argv[1] if len(sys.argv)>1 else "xpof"
D=np.load(f"models/m5xp_{MODE}_diag.npz")
WINS=["val","test24","test25","oos"]

def win(w):
    d={}
    for key in D.files:
        if key=="frozen": continue
        ww,_,k=key.partition("__")
        if ww==w: d[k]=D[key]
    return d
W={w:win(w) for w in WINS}

def nonoverlap(ts,mask,gap=300):
    take=[]; block=-1
    for i in np.where(mask)[0]:
        if ts[i]<block: continue
        take.append(i); block=int(ts[i])+gap
    return np.array(take,dtype=int)

def boot(corr,nb=4000,seed=7):
    corr=np.asarray(corr,float)
    if len(corr)<5: return (float("nan"),float("nan"))
    rng=np.random.default_rng(seed); n=len(corr)
    a=np.array([corr[rng.integers(0,n,n)].mean() for _ in range(nb)])
    return float(np.percentile(a,2.5)),float(np.percentile(a,97.5))

def gate_mask(d, conf_thr, agree_min=None, of_confirm=False, disp_max=None, comp_max=None):
    pr=d["pr"]; ny=d["ny"].astype(bool); conf=np.abs(pr-0.5)
    m=ny&(conf>=conf_thr)
    if agree_min is not None and "agree5" in d: m=m&(d["agree5"]>=agree_min)
    if disp_max is not None and "disp5" in d: m=m&(d["disp5"]<=disp_max)
    if comp_max is not None and "comp60" in d: m=m&(d["comp60"]<=comp_max)
    if of_confirm and "OF_of_norm_5" in d: m=m&(np.sign(d["OF_of_norm_5"])==np.sign(pr-0.5))
    return m

def evalg(d, **kw):
    pr=d["pr"]; y=d["y"]; ts=d["ts"].astype("int64")
    m=gate_mask(d,**kw); sel=nonoverlap(ts,m)
    if len(sel)==0: return (0,float("nan"),np.array([]))
    corr=((pr[sel]>0.5).astype(int)==y[sel]).astype(float)
    return (len(sel),float(corr.mean()),corr)

def yr(ts): return (np.asarray(ts,dtype="datetime64[s]").astype("datetime64[Y]").astype(int)+1970)

# ---------- Part A: descriptive conditional accuracy (NY + conf>=0.06) ----------
print(f"### Part A — descriptive conditional accuracy (mode={MODE}, NY-gated, non-overlap, conf>=0.06)")
base_thr=0.06
for cond_name, cond_fn in [
    ("agree5>=hi(70pct VAL)", None),
    ("OF_norm5 confirms dir", None),
    ("disp5<=lo(med VAL)", None),
]:
    pass
# compute VAL-derived cut points
va=W["val"]
agree_hi=float(np.nanpercentile(va["agree5"],70)) if "agree5" in va else None
disp_lo=float(np.nanpercentile(va["disp5"],50)) if "disp5" in va else None
comp_lo=float(np.nanpercentile(va["comp60"],50)) if "comp60" in va else None
print(f"VAL cuts: agree5>= {agree_hi:.3f} (70pct) | disp5<= {disp_lo:.2e} (med) | comp60<= {comp_lo:.2e} (med)")
def row(name, **kw):
    cells=[]
    for w in WINS:
        n,a,_=evalg(W[w],conf_thr=base_thr,**kw); cells.append(f"{w}:{a:.3f}(n{n})")
    print(f"  {name:<26} "+"  ".join(cells))
row("NY+conf only")
row("+agree5 high", agree_min=agree_hi)
row("+OF confirms", of_confirm=True)
row("+low disp5", disp_max=disp_lo)
row("+low comp60", comp_max=comp_lo)
row("+agree&OF&lowdisp", agree_min=agree_hi, of_confirm=True, disp_max=disp_lo)

# ---------- Part B: honest gate selection on VAL, report held-out floor ----------
print(f"\n### Part B — honest gate selection on VAL -> held-out (floor = min over test24/test25/oos)")
conf_grid=[float(np.nanpercentile(np.abs(va["pr"]-0.5)[va["ny"].astype(bool)],100*(1-c))) for c in (0.12,0.08,0.05,0.03,0.02,0.01)]
agree_grid=[None]+([float(np.nanpercentile(va["agree5"],q)) for q in (50,70,85)] if "agree5" in va else [])
disp_grid=[None]+([float(np.nanpercentile(va["disp5"],q)) for q in (50,30)] if "disp5" in va else [])
of_grid=[False,True]
combos=[]
for ct in conf_grid:
    for am in agree_grid:
        for dm in disp_grid:
            for oc in of_grid:
                combos.append(dict(conf_thr=ct,agree_min=am,disp_max=dm,of_confirm=oc))
# evaluate each combo on VAL and VAL halves
vts=va["ts"].astype("int64"); vy=yr(vts)
def val_eval(c):
    n,a,corr=evalg(va,**c)
    # split by VAL year halves for stability
    halves={}
    for yy in sorted(set(vy.tolist())):
        d2={k:(va[k][vy==yy] if hasattr(va[k],"__len__") and len(va[k])==len(vy) else va[k]) for k in va}
        nn,aa,_=evalg(d2,**c); halves[yy]=(nn,aa)
    return n,a,halves
scored=[]
for c in combos:
    n,a,halves=val_eval(c)
    if n<150: continue
    hvals=[v[1] for v in halves.values() if v[0]>=50]
    hmin=min(hvals) if hvals else float("nan")
    scored.append((c,n,a,hmin,halves))
def heldout(c):
    accs={}; corrs=[]
    for w in ("test24","test25","oos"):
        n,a,corr=evalg(W[w],**c); accs[w]=(n,a); corrs.append(corr)
    A=np.concatenate(corrs) if corrs else np.array([])
    lo,hi=boot(A); floor=min(accs[w][1] for w in accs)
    return accs,floor,(len(A),A.mean() if len(A) else float("nan"),lo,hi)
def desc(c):
    return f"conf{c['conf_thr']:.3f} ag{c['agree_min'] if c['agree_min'] is None else round(c['agree_min'],3)} dp{c['disp_max'] if c['disp_max'] is None else format(c['disp_max'],'.1e')} of{int(c['of_confirm'])}"
# B1: VAL-max
b1=max(scored,key=lambda z:z[2])
accs,floor,comb=heldout(b1[0])
print(f"[VAL-max]   {desc(b1[0])}  VAL acc={b1[2]:.3f}(n{b1[1]}) -> "+" ".join(f"{w}:{accs[w][1]:.3f}(n{accs[w][0]})" for w in accs)+f"  FLOOR={floor:.3f}  COMB n{comb[0]} {comb[1]:.3f} CI[{comb[2]:.3f},{comb[3]:.3f}]")
# B2: transfer-robust = require both VAL halves >=0.56 (n>=60), maximize the MIN half (stability)
robust=[z for z in scored if not np.isnan(z[3]) and z[3]>=0.56]
if robust:
    b2=max(robust,key=lambda z:z[3])
    accs,floor,comb=heldout(b2[0])
    print(f"[robust]    {desc(b2[0])}  VAL acc={b2[2]:.3f}(n{b2[1]}) halfmin={b2[3]:.3f} -> "+" ".join(f"{w}:{accs[w][1]:.3f}(n{accs[w][0]})" for w in accs)+f"  FLOOR={floor:.3f}  COMB n{comb[0]} {comb[1]:.3f} CI[{comb[2]:.3f},{comb[3]:.3f}]")
else:
    print("[robust]    no combo with both VAL halves >=0.56")
# B3: the combo with the best HELD-OUT floor (ORACLE — upper bound, NOT honest; tells us if 0.65 is even reachable)
best_floor=None
for c,n,a,hmin,halves in scored:
    accs,floor,comb=heldout(c)
    if comb[0]<150: continue
    if best_floor is None or floor>best_floor[1]: best_floor=(c,floor,accs,comb)
if best_floor:
    c,floor,accs,comb=best_floor
    print(f"[ORACLE max-floor, NOT honest] {desc(c)} -> "+" ".join(f"{w}:{accs[w][1]:.3f}(n{accs[w][0]})" for w in accs)+f"  FLOOR={floor:.3f}  (upper bound on what's reachable)")

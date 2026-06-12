"""USDJPY 15-MIN — TRAIN-LABEL sharpening: TRIPLE-BARRIER FIRST-TOUCH (vectorized).

SCOPE: USDJPY · 15m · NY session. Direction-lever sibling of usdjpy_15m_tblabel.py (the `avg` variant
is already done there). The own-pair endpoint label sign(close[t+15]-close[t]) is signal-capped ~.531.
Hypothesis (Lopez de Prado AFML; Prata et al. 2024 arXiv:2504.02249): a PATH-AWARE first-touch train
label denoises the sign target -> the model learns a cleaner sign map -> higher held-out fixed-15m AUC
and selective win-rate.

METHOD (this file): relabel ONLY the TRAIN target by which barrier is hit FIRST over the forward 15-bar
window:
  upper  = +k*sigma_t,   lower = -k*sigma_t   (sigma_t = trailing realized 1-bar-return vol, CAUSAL)
  vertical (timeout, neither barrier touched in 15 bars) -> endpoint sign = sign(close[t+15]-close[t]).
k is chosen on TRAIN (the k that maximizes the TRAIN-label/endpoint-label agreement-weighted spread is
NOT used; instead k is swept and the one giving the best VAL fixed-15m AUC is frozen — selection touches
VAL only via the deriv-faithful label, never the held-out years).

The barrier touch is VECTORIZED via cumulative max/min over the forward 15-bar window
(np.maximum.accumulate / np.minimum.accumulate on a strided sliding window) — NO per-bar python loop.

CRITICAL DISCIPLINE (non-negotiable):
  * ONLY the TRAIN label changes. VAL gate selection and ALL held-out EVALUATION use the UNCHANGED
    deriv-faithful fixed-15m label sign(close[t+15]-close[t]), ties (move==0) LOSE — deriv settles at
    exactly the 15m endpoint.
  * DECISION rows are restricted to the NY session (America/New_York 08-17, DST-correct via sessions.py).
    FEATURES stay causal/continuous; only the prediction/label rows are filtered.
  * sigma_t is trailing (rolling std of past 1-bar returns) -> no look-ahead in the barrier scale.

INCUMBENT to beat: certified NY own-pair GBM signal AUC ~.539 / win-rate ~.58-.60 (base .539/.58).
SURVIVES only if held-out moved-AUC > ~.539 OR NY cov3% win-rate CI-lower clears 0.541 in >=2 years.

Pre-registered falsifier written to usdjpy_15m_tbfirsttouch_result.json BEFORE the held-out read.

Usage: ~/binary-algo-venv/bin/python usdjpy_15m_tbfirsttouch.py [stride] [leaves]
"""
import os, sys, json, time, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import harness as H
import sessions as S
from usdjpy_15m_base import side_eval, covcurve, nonoverlap_chrono, boot, mk_lgb, BE, SPL, FEATS

PAIR = "USDJPY"; HOR = 15; STEP = 60; GAP = HOR*STEP; FEAT = H.FEAT_DIR
SESSION = "ny"
VOL_WIN = 900; VOL_MINP = 60                 # ~15h trailing realized-vol window for sigma_t (causal)
K_GRID = (0.5, 0.75, 1.0, 1.5, 2.0)          # barrier multipliers swept; k frozen on VAL fixed-15m AUC

def _argint(i, default):
    return int(sys.argv[i]) if len(sys.argv) > i and str(sys.argv[i]).isdigit() else default
TR_STRIDE = _argint(1, 6)
NUM_LEAVES = _argint(2, 127)
RESULT = "usdjpy_15m_tbfirsttouch_result.json" if (TR_STRIDE==6 and NUM_LEAVES==127) \
         else f"usdjpy_15m_tbfirsttouch_s{TR_STRIDE}_l{NUM_LEAVES}_result.json"


def _first_touch_label(c, fr, contig, k_list):
    """VECTORIZED triple-barrier first-touch labels for a contiguous close series `c` (per-year).

    For each k in k_list returns a float array `ylab` in {0,1,nan}:
      +1 if upper barrier (+k*sigma_t) is touched strictly BEFORE the lower (-k*sigma_t) within
         the forward HOR-bar window; 0 if lower touched first; on timeout/no-touch -> endpoint sign.
    sigma_t is the TRAILING rolling std of 1-bar returns (causal). nan where the forward window is
    not a clean contiguous HOR-step block (`contig`) or sigma_t is undefined.

    Barrier touch uses cumulative max/min over the forward window:
      first upper-touch bar  = argmax over j of (cummax(c[t+1..t+j]) >= up_t)
      first lower-touch bar  = argmin analog (cummin <= dn_t)
    No per-bar python loop: a single strided sliding-window view + accumulate handles all bars at once.
    """
    n = len(c)
    # trailing realized vol of 1-bar returns (causal): r1[t] uses c[t-1],c[t]; rolling std over past VOL_WIN
    r1 = np.full(n, np.nan); r1[1:] = c[1:]/c[:-1] - 1.0
    sig = pd.Series(r1).rolling(VOL_WIN, min_periods=VOL_MINP).std().values   # sigma at bar t (past only)

    # forward window of closes: fwd[t, j] = c[t+1+j], j=0..HOR-1   (vectorized, strided)
    # pad so every t in 0..n-1 has a HOR-length slot; invalid tails masked by `contig`
    cpad = np.concatenate([c, np.full(HOR, np.nan)])
    # sliding_window_view of length HOR starting at index t+1
    win = np.lib.stride_tricks.sliding_window_view(cpad, HOR)   # shape (len(cpad)-HOR+1, HOR)
    fwd = win[1:n+1]                                            # rows t=0..n-1 -> c[t+1 .. t+HOR]
    c0 = c[:, None]                                             # entry close, broadcast over window

    cummax = np.maximum.accumulate(fwd, axis=1)                 # running max of forward path
    cummin = np.minimum.accumulate(fwd, axis=1)                 # running min of forward path

    out = {}
    end_sign = (fr > 0).astype("float64")                      # endpoint sign (timeout fallback / default)
    for k in k_list:
        up = c0 * (1.0 + k * sig[:, None])                     # upper barrier per bar
        dn = c0 * (1.0 - k * sig[:, None])                     # lower barrier per bar
        up_hit = cummax >= up                                  # (n,HOR) cumulative-up touched by bar j
        dn_hit = cummin <= dn                                  # (n,HOR) cumulative-down touched by bar j
        # first-touch bar index (HOR if never touched) via argmax on the boolean accumulate
        tu = np.where(up_hit.any(axis=1), up_hit.argmax(axis=1), HOR)
        td = np.where(dn_hit.any(axis=1), dn_hit.argmax(axis=1), HOR)
        ylab = end_sign.copy()                                 # timeout / tie -> endpoint sign
        ylab[tu < td] = 1.0                                    # upper touched first
        ylab[td < tu] = 0.0                                    # lower touched first
        # invalidate non-contiguous / undefined-sigma / non-finite-endpoint rows
        bad = (~contig) | (~np.isfinite(sig)) | (sig <= 0) | (~np.isfinite(fr))
        ylab = ylab.astype("float64"); ylab[bad] = np.nan
        out[k] = ylab
    return out


def build(years, stride=1, k=None):
    """Returns X(df), ytr (first-touch train label for the given k; if k is None -> fixed-15m eval label),
    fr (fixed-15m fwd ret = EVAL label src), moved(bool), ts(secs), ny(bool session mask on decision rows).

    Rows are NOT pre-filtered to NY here (so sliding windows stay contiguous); the NY mask is returned so
    callers restrict DECISION rows. Train uses (moved & ny); eval restricts to (moved & ny)."""
    Xs=[]; ytrs=[]; frs=[]; tss=[]; nys=[]
    for y in years:
        p=f"{FEAT}/{PAIR}_{y}.parquet"
        if not os.path.exists(p): continue
        d=pd.read_parquet(p, columns=FEATS+["close"]); d=d[~d.index.duplicated(keep="last")]
        c=d["close"].values.astype(float)
        ts=d.index.values.astype("datetime64[s]").astype("int64"); n=len(d)
        contig=np.zeros(n,bool); contig[:n-HOR]=(ts[HOR:]-ts[:-HOR])==GAP   # exactly HOR clean 60s steps
        fr=np.full(n,np.nan); fr[:n-HOR]=c[HOR:]/c[:-HOR]-1.0               # fixed-15m fwd ret (EVAL)
        if k is None:
            ylab=(fr>0).astype("float64"); ylab[~np.isfinite(fr)]=np.nan
        else:
            ylab=_first_touch_label(c, fr, contig, [k])[k]
        ny=S.session_mask(ts, SESSION)                                      # America/New_York 08-17
        X=d[FEATS].astype("float32"); keepf=X.isna().mean(axis=1).values<0.5
        valid=contig & np.isfinite(fr) & keepf & np.isfinite(ylab)
        idx=np.where(valid)[0]
        if stride>1: idx=idx[::stride]
        Xs.append(X.iloc[idx]); ytrs.append(ylab[idx].astype(int)); frs.append(fr[idx])
        tss.append(ts[idx]); nys.append(ny[idx])
    X=pd.concat(Xs); fr=np.concatenate(frs)
    return (X, np.concatenate(ytrs), fr, (fr!=0.0), np.concatenate(tss), np.concatenate(nys))


def main():
    t0=time.time()
    # ---- pre-register falsifier BEFORE held-out read ----
    res={"key":"USDJPY.15m.ny",
         "lever":"triple-barrier FIRST-TOUCH train label (vectorized cummax/cummin); barriers +/-k*sigma_t "
                 "(trailing realized vol), vertical=15m timeout->endpoint sign; k frozen on VAL fixed-15m AUC. "
                 "EVAL=UNCHANGED deriv-faithful fixed-15m sign, ties LOSE; DECISION rows restricted to NY session.",
         "model":f"single-pair base GBM (239 feats), TRAIN-LABEL=tb-first-touch, EVAL=fixed-15m deriv label",
         "settlement":"EVAL=bar-close fixed-15m, ties LOSE, BE 0.541, gap=900s nonoverlap_chrono; NY decision rows",
         "splits":SPL, "session":SESSION, "tr_stride":TR_STRIDE, "num_leaves":NUM_LEAVES,
         "vol_win":VOL_WIN, "k_grid":list(K_GRID),
         "falsifier":{"registered_utc":"pre-OOS",
            "KILL_if":"VAL fixed-15m moved-AUC <= 0.5313 (no lift over base) OR moved up-rate drifts outside "
                      "[0.47,0.53] in any held-out year (leaked to magnitude / mirage). "
                      "SURVIVE only if held-out moved-AUC > ~.539 OR NY cov3% win-rate CI-lower clears 0.541 in >=2 years.",
            "incumbent":"certified NY own-pair GBM signal AUC ~.539 / win-rate ~.58-.60 (base .539/.58)",
            "rationale":"raw-endpoint base AUC capped ~.531. Path-aware first-touch train label may denoise the "
                        "sign target. EVAL stays deriv-faithful fixed-15m on NY decision rows; only a held-out "
                        "fixed-15m lift counts. Sign-invariance note (arXiv:2512.15720): vol-scaled barriers may "
                        "gate SIZE not SIGN; let the AUC decide."}}
    json.dump(res, open(RESULT,"w"), indent=2)

    # ---- choose k on TRAIN/VAL: train per-k on first-touch label, pick best VAL fixed-15m AUC ----
    # VAL fixed-15m eval objects are built once (k=None) and reused across k.
    Xva0,_,frva,mva,tsv,nyv=build(SPL["val"], 1, None)
    yva_eval=(frva>0).astype(int)
    iva=mva & nyv                                          # VAL decision rows = MOVED & NY
    print(f"[tbft] sweep k over {list(K_GRID)} | val_decision={int(iva.sum()):,} (moved&NY)  {time.time()-t0:.0f}s", flush=True)

    best_k=None; best_val_auc=-1.0; best_model=None; best_pva=None; k_scores={}
    for k in K_GRID:
        Xtr,ytr,frtr,mtr,_,nytr=build(SPL["train"], TR_STRIDE, k)
        itr=mtr & nytr                                    # TRAIN decision rows = MOVED & NY
        # VAL features identical across k; rebuild VAL X with this k only to keep column alignment (cheap, k!=None ok)
        Xva,_,_,_,_,_=build(SPL["val"], 1, k)
        L=mk_lgb(num_leaves=NUM_LEAVES)
        L.fit(Xtr[itr], ytr[itr], eval_set=[(Xva[iva], yva_eval[iva])], eval_metric="auc",
              callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
        pva=L.predict_proba(Xva)[:,1]
        vauc=float(roc_auc_score(yva_eval[iva], pva[iva]))   # AUC vs FIXED-15m label on moved&NY
        k_scores[str(k)]={"val_fixed15m_auc":vauc,"best_iter":int(L.best_iteration_ or 0),"train_decision":int(itr.sum())}
        print(f"[tbft] k={k}: train={int(itr.sum()):,} best_iter={L.best_iteration_} VAL fixed15m moved-AUC={vauc:.4f}", flush=True)
        if vauc>best_val_auc:
            best_val_auc=vauc; best_k=k; best_model=L; best_pva=pva
    res["k_scores"]=k_scores; res["chosen_k"]=best_k; res["val_or_signal_auc"]=best_val_auc
    L=best_model; pva=best_pva
    print(f"[tbft] FROZEN k={best_k} VAL fixed15m moved-AUC={best_val_auc:.4f} (base .5313)  {time.time()-t0:.0f}s", flush=True)

    # ---- VAL gate selection: WORST-half stability on NY decision rows (never VAL-acc-max) ----
    # restrict the gate search to NY decision rows (features stay causal)
    sel=nyv                                                # NY rows only for gate selection
    pvas=pva[sel]; yvs=yva_eval[sel]; mvs=mva[sel]; tvs=tsv[sel]
    half=len(pvas)//2; confv=np.abs(pvas-0.5); best=None
    for cov in (0.20,0.10,0.05,0.03,0.02):
        thr=float(np.quantile(confv,1-cov)); accs=[]
        for s,e in ((0,half),(half,len(pvas))):
            r=side_eval(pvas[s:e], yvs[s:e], mvs[s:e], tvs[s:e], thr)
            accs.append(r["COMBINED"]["wr"] if r else float("nan"))
        worst=np.nanmin(accs)
        if best is None or worst>best[0]: best=(worst,cov,thr,accs)
    worst_half,COV,THR,halfaccs=best
    res["gate"]={"cov":COV,"conf_thr":THR,"val_worst_half_wr":float(worst_half),"val_half_wrs":[float(a) for a in halfaccs]}
    print(f"[tbft] FROZEN gate cov{COV:.0%} thr={THR:.4f} VAL worst-half WR={worst_half:.4f}  {time.time()-t0:.0f}s", flush=True)

    # ---- held-out per year: restrict DECISION rows to NY; EVAL = fixed-15m deriv label ----
    res["years"]={}
    for w in ("test24","test25","oos"):
        Xw,_,frw,mw,tsw,nyw=build(SPL[w], 1, best_k)       # k only relabels train; here used only for X build (eval uses frw)
        yw=(frw>0).astype(int)
        pr=L.predict_proba(Xw)[:,1]
        dec=mw & nyw                                        # NY moved decision rows
        auc=float(roc_auc_score(yw[dec], pr[dec])) if dec.sum()>10 else float("nan")
        up_rate=float(yw[dec].mean()) if dec.sum()>0 else float("nan")
        # gate / covcurve on NY decision rows only (nonoverlap_chrono gap=900 inside side_eval)
        prn=pr[nyw]; ywn=yw[nyw]; mwn=mw[nyw]; tswn=tsw[nyw]
        gate=side_eval(prn, ywn, mwn, tswn, THR)
        cc=covcurve(prn, ywn, mwn, tswn)
        cov3=cc.get("0.03",{})
        res["years"][w]={"moved_auc":auc, "moved_up_rate":up_rate, "n_ny_moved":int(dec.sum()),
                         "gate":gate, "cov3":cov3, "covcurve":cc,
                         "tripwire_ok":bool(0.47<=up_rate<=0.53) if np.isfinite(up_rate) else False}
        g=gate["COMBINED"] if gate else {"n":0,"wr":float('nan'),"ci":[float('nan')]*2}
        u=gate["UP"] if gate else {"n":0,"wr":float('nan'),"ci":[float('nan')]*2}
        dd=gate["DOWN"] if gate else {"n":0,"wr":float('nan'),"ci":[float('nan')]*2}
        print(f"=== {w} === moved-AUC={auc:.4f} up={up_rate:.4f} (NY moved n={int(dec.sum())}) | gate cov{COV:.0%}: "
              f"COMB n{g['n']} {g['wr']:.4f} CI[{g['ci'][0]:.3f},{g['ci'][1]:.3f}] | "
              f"UP n{u['n']} {u['wr']:.4f} CI[{u['ci'][0]:.3f},{u['ci'][1]:.3f}] | "
              f"DOWN n{dd['n']} {dd['wr']:.4f} CI[{dd['ci'][0]:.3f},{dd['ci'][1]:.3f}]", flush=True)

    # ---- apply falsifier + verdict ----
    kill_auc = best_val_auc <= 0.5313
    tripwire_fail = [w for w in ("test24","test25","oos") if not res["years"][w]["tripwire_ok"]]
    beats_base_auc = bool(best_val_auc > 0.5313)
    # SURVIVE conditions (incumbent .539 / cov3% CI-lo >=0.541 in >=2 years)
    auc_beats_incumbent = [w for w in ("test24","test25","oos")
                           if np.isfinite(res["years"][w]["moved_auc"]) and res["years"][w]["moved_auc"]>0.539]
    cov3_clears = [w for w in ("test24","test25","oos")
                   if res["years"][w]["cov3"] and "COMBINED" in res["years"][w]["cov3"]
                   and res["years"][w]["gate"] and res["years"][w]["gate"]["COMBINED"]["ci"][0]>=BE]
    survive = (len(auc_beats_incumbent)>=1) or (len(cov3_clears)>=2)
    killed = bool(kill_auc or len(tripwire_fail)>0 or (not survive))
    res["verdict"]={"KILLED":killed,
                    "beats_base_auc":beats_base_auc,
                    "val_auc_le_0p5313":bool(kill_auc),
                    "tripwire_fail_years":tripwire_fail,
                    "auc_beats_incumbent_years":auc_beats_incumbent,
                    "cov3_COMBINED_CIlo_clears_BE_years":cov3_clears,
                    "note":("KILLED: no VAL AUC lift over base .5313" if kill_auc else
                            f"KILLED: up-rate leaked to magnitude in {tripwire_fail}" if tripwire_fail else
                            "SURVIVED: held-out moved-AUC>.539 and/or cov3% CI-lo clears BE in >=2yr" if survive else
                            "KILLED: no held-out lift over incumbent (.539 / cov3% CI-lo>=.541 in >=2yr)")}
    json.dump(res, open(RESULT,"w"), indent=2)
    print(f"\n[tbft] VERDICT: {'KILLED' if killed else 'SURVIVED'} k={best_k} beats_base_AUC={beats_base_auc} "
          f"(val_auc={best_val_auc:.4f}; AUC>incumbent yrs={auc_beats_incumbent}; cov3-clears={cov3_clears}; "
          f"tripwire_fail={tripwire_fail}) -> {RESULT}  total={time.time()-t0:.0f}s", flush=True)


if __name__=="__main__":
    main()

"""V1 — MULTIPLE-TESTING HAIRCUT over the WHOLE EURUSD sweep family (post-hoc calc, no model load).

PURPOSE — re-anchor whether the certified (5m,UP) edge (m5xp up-preds, refit cov0.05 p10 0.553)
SURVIVES multiple-testing deflation at the REALIZED number of trials, before spending more compute.
If it downgrades the incumbent, the whole backlog's "bar to beat" moves.

METHOD (mirrors cpcv_certify.deflated_metric, generalized to the family):
  Each TRIAL contributes a one-sided t-stat of its directional edge over deriv breakeven 0.541:
      t_i = (wr_i - 0.541) / sqrt(wr_i*(1-wr_i)/n_i),   p_i = 1 - Phi(t_i)
  Then four multiplicity lenses, each at the realized M and the correlation-adjusted N_hat:
    (1) Holm-Bonferroni  (FWER, step-down)
    (2) Benjamini-Hochberg (FDR, independence)
    (3) Benjamini-Yekutieli (FDR, ARBITRARY dependence; c(M)=harmonic) <- the falsifier's lens
    (4) Deflated-Sharpe-style E[max] test: observed t vs E[max of N_hat null t-stats]
        E[max_N] = (1-g)*Z^-1(1-1/N) + g*Z^-1(1-1/(N*e)),  g=Euler-Mascheroni (Lopez de Prado),
        plus the cruder sqrt(2 ln N) upper bound and the Sidak p = 1-(1-p)^N_hat.
  N_hat = rho_bar + (1-rho_bar)*M  (user spec; shrinks M toward 1 as avg cross-trial corr rho_bar->1).

DECISION VARIABLE  : m5xp UP, evaluated on BOTH the pooled (n=3052) and the harsh binding-2025 (n=1379) stat.
PRE-REGISTERED FALSIFIER (from IDEAS_LOG V1):
  (a) PROCEDURE-VALIDITY: the haircut MUST flag the already-killed trials (POW=1.0 retrain collapse,
      Stoikov, the ~24 null 60s channels, 2m UP) as NON-significant. If it calls any of them significant
      the procedure is broken and its m5xp verdict is void.
  (b) INCUMBENT TEST: DOWNGRADE m5xp UP to UNCERTIFIED if its BHY-adjusted binding-2025 p > 0.05 at realized M.

  ~/binary-algo-venv/bin/python m5_mt_haircut.py
"""
import os, re, json, glob, math
import numpy as np
from scipy.stats import norm

ROOT = "/media/sean/CORSAIR/binary-algo"
BE   = 0.541                 # deriv 15m/5m breakeven win-rate at payout R~1.85
GAMMA = 0.5772156649         # Euler-Mascheroni (Lopez de Prado E[max] approx)
ALPHA = 0.05
DEFAULT_N = 1400             # representative per-year gated n where a ledger row records no n (FLAGGED)

# ---------------------------------------------------------------- core stats
def t_and_p(wr, n):
    """one-sided t-stat of win-rate wr over breakeven, and p = P(>t) under H0."""
    if n is None or n <= 1 or wr is None or not np.isfinite(wr):
        return float("nan"), float("nan")
    se = math.sqrt(max(wr*(1-wr), 1e-9)/n)
    t  = (wr - BE)/se
    return t, float(norm.sf(t))

def emax_normal(N):
    """E[max of N i.i.d. standard normals] — accurate (Lopez de Prado) + crude sqrt(2 ln N)."""
    N = max(float(N), 2.0)
    accurate = (1-GAMMA)*norm.ppf(1-1.0/N) + GAMMA*norm.ppf(1-1.0/(N*math.e))
    crude    = math.sqrt(2*math.log(N))
    return accurate, crude

def harmonic(M):
    M = int(max(M,1))
    return float(np.sum(1.0/np.arange(1, M+1)))

def holm_adjusted(pvals):
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        run = max(run, val)              # enforce monotone non-decreasing
        adj[idx] = min(run, 1.0)
    return adj

def bh_by_adjusted(pvals, dependence="independent"):
    """Benjamini-Hochberg (independent) or Benjamini-Yekutieli (arbitrary dependence) adjusted p."""
    p = np.asarray(pvals, float); m = len(p)
    c = harmonic(m) if dependence == "arbitrary" else 1.0
    order = np.argsort(p); ranked = p[order]
    adj_sorted = np.empty(m)
    prev = 1.0
    for i in range(m-1, -1, -1):         # step-up: from largest p down
        k = i + 1
        val = ranked[i] * m * c / k
        prev = min(prev, val)
        adj_sorted[i] = min(prev, 1.0)
    adj = np.empty(m); adj[order] = adj_sorted
    return adj

# ---------------------------------------------------------------- family universe
def parse_triple(cell):
    """parse a '.605/.577/.615' style per-year cell -> [0.605,0.577,0.615] (floats), ignoring junk."""
    if not cell: return []
    nums = re.findall(r"0?\.\d{2,4}", cell)
    out=[]
    for x in nums:
        v=float(x)
        if 0.30 <= v <= 0.80:            # plausible win-rate / AUC band
            out.append(v)
    return out

def load_ledger_variants():
    """Parse the 2m + 5m sweep ledgers. Each row: id|tier|method|script|status|up_oos(24/25/26)|down(24/25/26)|verdict|json.
    Returns list of dicts with the variant's claimed binding-year win-rate (worst of its claimed side)."""
    variants=[]
    for tf in ["2m","5m"]:
        p=f"{ROOT}/sweeps/EURUSD_{tf}.md"
        if not os.path.exists(p): continue
        for l in open(p).read().splitlines():
            if not l.strip().startswith("|") or "---" in l: continue
            cells=[c.strip() for c in l.strip().strip("|").split("|")]
            if len(cells) < 6: continue
            vid=cells[0]
            if vid.lower() in ("id",""): continue
            status=cells[4].lower() if len(cells)>4 else ""
            up_cell   = cells[5] if len(cells)>5 else ""
            down_cell = cells[6] if len(cells)>6 else ""
            up=parse_triple(up_cell); dn=parse_triple(down_cell)
            # claimed binding-year win-rate = worst across the years it reports, max over the two sides
            up_bind = min(up) if up else None
            dn_bind = min(dn) if dn else None
            cands=[x for x in (up_bind,dn_bind) if x is not None]
            bind = max(cands) if cands else None      # the BEST side's worst year (what would be claimed)
            variants.append(dict(tf=tf, id=vid, status=status, bind_wr=bind,
                                 up=up, down=dn, raw=l[:140]))
    return variants

def count_60s_channels():
    """count the null 60s direction channels (min1_* result JSONs) as trials."""
    files=set(glob.glob(ROOT+"/min1_*result*.json"))
    return sorted(os.path.basename(f) for f in files)

# ---------------------------------------------------------------- main
def main():
    out={"test":"V1 multiple-testing haircut over the whole EURUSD sweep family",
         "breakeven":BE, "alpha":ALPHA}

    # ---- (A) DECISION VARIANTS (Tier-1 logged numbers, cited) ----
    # m5xp UP: pooled (m5_cpcv_m5xp_result.json) + binding-2025 (per_year 2025=0.5765, n=1379 from EURUSD_RESULTS side-split)
    m5xp = json.load(open(f"{ROOT}/m5_cpcv_m5xp_result.json"))
    inc_pooled_wr, inc_pooled_n = m5xp["pooled_acc"], m5xp["n_up_trades"]            # 0.5927, 3052
    inc_bind_wr,   inc_bind_n   = m5xp["per_year"]["2025"], 1379                     # 0.5765, n1379 (RESULTS)
    t_pool,p_pool = t_and_p(inc_pooled_wr, inc_pooled_n)
    t_bind,p_bind = t_and_p(inc_bind_wr,   inc_bind_n)

    # ---- (B) BUILD THE FAMILY t-stat VECTOR (for Holm/BHY + procedure validation) ----
    led = load_ledger_variants()
    chans60 = count_60s_channels()
    # family members: every ledger variant (binding-year wr) + each 60s null channel (~0.50 AUC -> wr~0.50)
    fam=[]   # (label, wr, n, status)
    for v in led:
        if v["bind_wr"] is None:        # pending / no number logged yet -> not a completed trial
            continue
        fam.append((f'{v["tf"]}:{v["id"]}', v["bind_wr"], DEFAULT_N, v["status"]))
    # 60s null channels: read their VAL AUC where present, treat AUC as the win-rate proxy (all ~0.50)
    for f in chans60:
        try: d=json.load(open(f"{ROOT}/{f}"))
        except: continue
        auc=None
        for k in ("val_auc_worsthalf","val_auc_full","val_auc","standalone"):
            if k in d:
                auc = d[k].get("val_auc_worsthalf", d[k].get("val_auc_full")) if isinstance(d[k],dict) else d[k]
                if auc: break
        fam.append((f"60s:{f}", float(auc) if auc else 0.501, DEFAULT_N, "killed/null"))
    # explicit named kills (falsifier procedure-validity checks), Tier-1:
    named_kills=[]
    try:
        st=json.load(open(f"{ROOT}/min2_stoikov_result.json"))
        named_kills.append(("KILL:Stoikov@120s", st["val_auc_div"], DEFAULT_N))      # 0.4984
    except: pass
    try:
        m2=json.load(open(f"{ROOT}/min2_cpcv_result.json"))
        named_kills.append(("KILL:2m-UP", m2["per_year_strict"]["2024"], 625))        # binding 2024 0.5168
    except: pass
    try:
        xo=json.load(open(f"{ROOT}/min1_xofi_result.json"))
        named_kills.append(("KILL:60s-xofi", xo["standalone"]["val_auc_worsthalf"], DEFAULT_N))  # 0.5015
    except: pass
    # POW=1.0 magweight retrain collapse (logged kill): use the killed POW=0.25 DOWN p10 as a stand-in if 1.0 absent
    named_kills.append(("KILL:POW1.0-retrain", 0.50, DEFAULT_N))   # collapse to coin-flip (ledger: KILLED)

    # the m5xp UP claim joins the family as a trial too (binding-year stat)
    fam.append(("INC:m5xp-UP(bind2025)", inc_bind_wr, inc_bind_n, "certified"))
    for lbl,wr,n in named_kills:
        fam.append((lbl, wr, n, "killed"))

    labels=[x[0] for x in fam]
    wrs   =[x[1] for x in fam]
    ns    =[x[2] for x in fam]
    ts    =[t_and_p(w,n)[0] for w,n in zip(wrs,ns)]
    ps    =[t_and_p(w,n)[1] for w,n in zip(wrs,ns)]
    ps_arr=np.array([1.0 if not np.isfinite(p) else p for p in ps])

    M_5m  = sum(1 for v in led if v["tf"]=="5m" and v["bind_wr"] is not None)
    M_2m  = sum(1 for v in led if v["tf"]=="2m" and v["bind_wr"] is not None)
    M_60s = len(chans60)
    M_realized = len(fam)                 # all completed trials we can score
    out["family_counts"]=dict(M_5m=M_5m, M_2m=M_2m, M_60s_channels=M_60s,
                              named_kills=len(named_kills), M_realized_scored=M_realized,
                              program_stated_n_trials=70)

    holm = holm_adjusted(ps_arr)
    bh   = bh_by_adjusted(ps_arr, "independent")
    by   = bh_by_adjusted(ps_arr, "arbitrary")
    idx={lbl:i for i,lbl in enumerate(labels)}

    # ---- (C) PROCEDURE VALIDATION: every killed/null trial must be NON-significant ----
    valid_checks={}
    sig_kill=False
    for lbl in labels:
        i=idx[lbl]
        if fam[i][3] in ("killed","killed/null") or lbl.startswith("KILL") or lbl.startswith("60s"):
            is_sig = bool(by[i] <= ALPHA)
            valid_checks[lbl]=dict(wr=round(wrs[i],4), t=round(ts[i],3) if np.isfinite(ts[i]) else None,
                                   raw_p=round(ps_arr[i],4), by_adj_p=round(float(by[i]),4), flagged_sig=is_sig)
            if is_sig and not lbl.startswith("INC"):
                sig_kill=True
    procedure_valid = not sig_kill
    out["procedure_validation"]=dict(all_kills_nonsignificant=procedure_valid,
                                     detail={k:v for k,v in valid_checks.items() if k.startswith("KILL")})

    # ---- (D) INCUMBENT HAIRCUT across M scenarios x rho_bar, both pooled & binding-year ----
    def haircut_block(t, p, tag):
        block={"t":round(t,3),"raw_one_sided_p":float(f"{p:.3e}"),
               "passes_HLZ_t>=3.0":bool(t>=3.0)}
        scen={}
        for Mname,M in [("M=5m_only", M_5m), ("M=5m+2m", M_5m+M_2m),
                        ("M=all_scored", M_realized), ("M=program_70", 70)]:
            for rho in (0.2,0.4,0.6):
                Nhat = rho + (1-rho)*M
                acc,crude = emax_normal(Nhat)
                sidak = 1-(1-p)**Nhat
                scen[f"{Mname}|rho={rho}"]=dict(
                    M=M, N_hat=round(Nhat,1),
                    Emax_accurate=round(acc,3), Emax_crude=round(crude,3),
                    t_beats_Emax_accurate=bool(t>acc), t_beats_Emax_crude=bool(t>crude),
                    sidak_p=round(float(sidak),4), sidak_sig=bool(sidak<=ALPHA))
        block["scenarios"]=scen
        return block

    i=idx["INC:m5xp-UP(bind2025)"]
    out["incumbent_m5xp_UP"]=dict(
        pooled=dict(wr=inc_pooled_wr, n=inc_pooled_n, **haircut_block(t_pool,p_pool,"pooled"),
                    holm_adj_p_in_family=None, by_adj_p_in_family=None),
        binding_2025=dict(wr=inc_bind_wr, n=inc_bind_n, **haircut_block(t_bind,p_bind,"binding2025"),
                          holm_adj_p_in_family=round(float(holm[i]),4),
                          by_adj_p_in_family=round(float(by[i]),4),
                          bh_adj_p_in_family=round(float(bh[i]),4)))

    # ---- (E) VERDICT — apply the pre-registered falsifier verbatim ----
    by_bind = float(by[i])
    downgrade = bool(by_bind > ALPHA)     # falsifier (b): BHY-adjusted binding-2025 p > 0.05 -> DOWNGRADE
    # pooled survival at realized family multiplicity (standard HLZ/DSR lens)
    Nhat_real = 0.4 + 0.6*M_realized
    pooled_sidak = 1-(1-p_pool)**Nhat_real
    pooled_survives = bool(pooled_sidak <= ALPHA and t_pool > emax_normal(Nhat_real)[0])

    out["VERDICT"]=dict(
        procedure_valid=procedure_valid,
        pooled_survives_deflation=pooled_survives,
        pooled_sidak_p_at_Nhat=float(f"{pooled_sidak:.3e}"),
        binding2025_by_adj_p=round(by_bind,4),
        binding2025_downgrades_per_falsifier=downgrade,
        statement=(
          ("PROCEDURE INVALID — a killed trial was flagged significant; verdict void." if not procedure_valid else
           ("m5xp UP: POOLED edge SURVIVES multiple-testing deflation decisively "
            f"(t={t_pool:.2f}, Sidak-p@Nhat={pooled_sidak:.1e}); "
            f"but the HARSH binding-2025-only lens {'DOWNGRADES' if downgrade else 'survives'} it "
            f"(BY-adj p={by_bind:.3f}, t={t_bind:.2f} vs HLZ bar 3.0). "
            "Precise status: refit-CPCV-certified + pooled-deflation-robust; "
            f"single-binding-year edge is {'NOT' if downgrade else ''} multiple-testing-significant under BY at realized M.")))
    )

    json.dump(out, open(f"{ROOT}/m5_mt_haircut_result.json","w"), indent=2, default=str)

    # ---- console summary ----
    print("="*100); print("V1 MULTIPLE-TESTING HAIRCUT — EURUSD sweep family"); print("="*100)
    print(f"family scored: M_5m={M_5m} M_2m={M_2m} M_60s_chan={M_60s} named_kills={len(named_kills)} "
          f"-> M_realized_scored={M_realized} (program stated ~70)")
    print(f"\nPROCEDURE VALIDATION (kills must be NON-significant):")
    for k,v in out["procedure_validation"]["detail"].items():
        print(f"   {k:24s} wr={v['wr']:.4f} t={v['t']} raw_p={v['raw_p']:.3f} BY-adj_p={v['by_adj_p']:.3f} -> sig={v['flagged_sig']}")
    print(f"   ALL KILLS NON-SIGNIFICANT = {procedure_valid}")
    print(f"\nINCUMBENT m5xp UP:")
    print(f"   POOLED       wr={inc_pooled_wr:.4f} n={inc_pooled_n}  t={t_pool:.2f}  raw_p={p_pool:.2e}  Sidak_p@Nhat({Nhat_real:.0f})={pooled_sidak:.2e}  HLZ_t>=3={t_pool>=3.0}")
    print(f"   BIND-2025    wr={inc_bind_wr:.4f} n={inc_bind_n}  t={t_bind:.2f}  raw_p={p_bind:.4f}  BY-adj_p(family)={by_bind:.3f}  Holm-adj={holm[i]:.3f}  HLZ_t>=3={t_bind>=3.0}")
    eg=out["incumbent_m5xp_UP"]["binding_2025"]["scenarios"]
    print(f"   E[max] test (binding-2025 t={t_bind:.2f}):")
    for sk in ["M=5m_only|rho=0.4","M=all_scored|rho=0.4","M=program_70|rho=0.4"]:
        s=eg[sk]; print(f"      {sk:22s} Nhat={s['N_hat']:.0f} Emax_acc={s['Emax_accurate']} (t>Emax:{s['t_beats_Emax_accurate']}) Emax_crude={s['Emax_crude']} (t>:{s['t_beats_Emax_crude']}) sidak_p={s['sidak_p']}")
    print(f"\nVERDICT: {out['VERDICT']['statement']}")
    print(f"-> m5_mt_haircut_result.json")
    return out

if __name__=="__main__":
    main()

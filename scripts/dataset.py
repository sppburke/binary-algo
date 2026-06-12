"""Unified dataset loader: base MTF features + cross-pair lead-lag + order-flow proxy.
Toggle each block to run clean ablations. Returns (X, y, aux) per split with row stride."""
import os, numpy as np, pandas as pd
import harness as H, crosspair as CP, orderflow as OF

FEAT_DIR=H.FEAT_DIR; OF_DIR=OF.OF_DIR
TARGET="EURUSD"

def feature_list(target=TARGET, use_cross=True, use_of=True, use_peer_of=False):
    feats=list(H.feature_cols(target))
    if use_cross: feats+=CP.cross_feature_names(target)
    if use_of: feats+=[f"SELF_{c}" for c in OF.of_feature_names()]
    if use_peer_of:
        for pr in [p for p in CP.ALL_PAIRS if p!=target]:
            feats+=[f"{pr[:6]}OF_{c}" for c in OF.of_feature_names()]
    return feats

def load_year(target, year, use_cross=True, use_of=True, use_peer_of=False):
    base_cols=list(H.feature_cols(target))+H.META_COLS
    if use_cross:
        df=CP.load_year_merged(target, year, base_cols=base_cols)
    else:
        p=f"{FEAT_DIR}/{target}_{year}.parquet"
        df=pd.read_parquet(p, columns=base_cols) if os.path.exists(p) else None
    if df is None: return None
    if use_of:
        ofp=f"{OF_DIR}/{target}_{year}.parquet"
        if os.path.exists(ofp):
            o=pd.read_parquet(ofp).reindex(df.index, method="ffill", limit=3)
            o.columns=[f"SELF_{c}" for c in o.columns]
            df=pd.concat([df,o],axis=1)
    if use_peer_of:
        for pr in [p for p in CP.ALL_PAIRS if p!=target]:
            ofp=f"{OF_DIR}/{pr}_{year}.parquet"
            if os.path.exists(ofp):
                o=pd.read_parquet(ofp).reindex(df.index, method="ffill", limit=3)
                o.columns=[f"{pr[:6]}OF_{c}" for c in o.columns]
                df=pd.concat([df,o],axis=1)
    return df

def load_split(split, target=TARGET, stride=1, use_cross=True, use_of=True, use_peer_of=False):
    feats=feature_list(target, use_cross, use_of, use_peer_of)
    parts=[]
    for y in H.SPLITS[split]:
        df=load_year(target,y,use_cross,use_of,use_peer_of)
        if df is None: continue
        df=df[df.valid==True]
        if stride>1: df=df.iloc[::stride]
        parts.append(df)
    df=pd.concat(parts)
    # ensure all feature cols exist
    for f in feats:
        if f not in df.columns: df[f]=np.nan
    X=df[feats].astype("float32")
    y=df["y"].astype(int).values
    aux=df[["y","fwd_ret","close"]].copy()
    return X,y,aux

if __name__=="__main__":
    X,y,aux=load_split("val", stride=20)
    print("val sample X",X.shape,"y_mean",round(float(y.mean()),4))
    print("n features",X.shape[1])
    print("nan frac",round(float(X.isna().mean().mean()),4))

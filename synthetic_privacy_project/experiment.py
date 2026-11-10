"""Preliminary experiment: do aggregate privacy checks hide record-level risk?

Fits several generators on a training split of UCI Adult, then measures
  * utility  - train-on-synthetic, test-on-real (TSTR) accuracy / F1
  * aggregate privacy check - share of synthetic records closer to train than holdout (DCR share)
  * record-level risk - a distance-based membership inference attack (MIA), overall
    and separately for outlier records vs. typical records.

Usage: python experiment.py --out results/ --seeds 0 1 2
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
COLS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
        "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
        "hours_per_week", "native_country", "income"]
NUM = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CAT = [c for c in COLS if c not in NUM]
TARGET = "income"
N_TRAIN, N_HOLDOUT, N_TEST = 5000, 5000, 10000
OUTLIER_FRAC = 0.10


def load_adult(path=None):
    df = pd.read_csv(path or DATA_URL, header=None, names=COLS, na_values="?",
                     skipinitialspace=True)
    df = df.dropna().reset_index(drop=True)
    df[TARGET] = df[TARGET].str.replace(".", "", regex=False)
    return df


class Encoder:
    """Standardize numerics and one-hot categoricals, fit on real training data only."""

    def fit(self, df):
        self.mu, self.sd = df[NUM].mean(), df[NUM].std().replace(0, 1)
        self.levels = {c: sorted(df[c].unique()) for c in CAT}
        return self

    def transform(self, df):
        parts = [((df[NUM] - self.mu) / self.sd).to_numpy()]
        for c in CAT:
            # one-hot scaled by 1/sqrt(2) so a category mismatch adds 1 to the squared distance
            parts.append((df[c].to_numpy()[:, None] == np.array(self.levels[c])[None, :])
                         .astype(float) / np.sqrt(2))
        return np.hstack(parts)


def interpolation_baseline(train, n, rng, k=5):
    """SMOTE-style generator for mixed data: interpolate numerics toward a same-class
    neighbor, copy each categorical from one of the two parents."""
    enc = Encoder().fit(train)
    X = enc.transform(train)
    rows = []
    for label, grp in train.groupby(TARGET):
        idx = grp.index.to_numpy()
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(idx))).fit(X[idx])
        m = int(round(n * len(idx) / len(train)))
        a = rng.choice(len(idx), m)
        nbr = nn.kneighbors(X[idx][a], return_distance=False)[:, 1:]
        b = nbr[np.arange(m), rng.integers(0, nbr.shape[1], m)]
        A, B = train.loc[idx[a]].reset_index(drop=True), train.loc[idx[b]].reset_index(drop=True)
        lam = rng.random(m)[:, None]
        out = A.copy()
        out[NUM] = A[NUM].to_numpy() + lam * (B[NUM].to_numpy() - A[NUM].to_numpy())
        for c in CAT:
            pick = rng.random(m) < 0.5
            out.loc[pick, c] = B.loc[pick, c]
        rows.append(out)
    out = pd.concat(rows, ignore_index=True)
    int_cols = ["age", "education_num", "hours_per_week", "fnlwgt", "capital_gain", "capital_loss"]
    out[int_cols] = out[int_cols].round()
    return out


def generate(name, train, n, seed):
    rng = np.random.default_rng(seed)
    if name == "Copy (reference)":
        return train.sample(n, replace=True, random_state=seed).reset_index(drop=True)
    if name == "Interpolation (SMOTE-style)":
        return interpolation_baseline(train, n, rng)
    md = Metadata.detect_from_dataframes({"adult": train})
    for c in CAT:
        md.update_column(c, sdtype="categorical")
    for c in NUM:
        md.update_column(c, sdtype="numerical")
    if name == "Gaussian copula":
        model = GaussianCopulaSynthesizer(md)
    elif name == "CTGAN":
        model = CTGANSynthesizer(md, epochs=300, verbose=False)
    elif name == "TVAE":
        model = TVAESynthesizer(md, epochs=300)
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.fit({"adult": train})
    syn = model.sample(n)
    return syn["adult"] if isinstance(syn, dict) else syn


def tstr(syn, test):
    X_cols = [c for c in COLS if c != TARGET]
    cats = [c for c in CAT if c != TARGET]
    Xs, Xt = syn[X_cols].copy(), test[X_cols].copy()
    for c in cats:
        levels = sorted(set(Xs[c]) | set(Xt[c]))
        Xs[c] = pd.Categorical(Xs[c], categories=levels)
        Xt[c] = pd.Categorical(Xt[c], categories=levels)
    ys, yt = (syn[TARGET] == ">50K").astype(int), (test[TARGET] == ">50K").astype(int)
    if ys.nunique() < 2:
        return float("nan"), float("nan")
    clf = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0)
    clf.fit(Xs, ys)
    p = clf.predict(Xt)
    return accuracy_score(yt, p), f1_score(yt, p)


def privacy(syn, train, holdout, enc, outlier_mask_members, outlier_mask_nonmembers):
    S = enc.transform(syn)
    Tr, Ho = enc.transform(train), enc.transform(holdout)
    # aggregate check: for each synthetic record, is its nearest real record in train or holdout?
    d_tr = NearestNeighbors(n_neighbors=1).fit(Tr).kneighbors(S)[0][:, 0]
    d_ho = NearestNeighbors(n_neighbors=1).fit(Ho).kneighbors(S)[0][:, 0]
    dcr_share = float(np.mean(d_tr < d_ho) + 0.5 * np.mean(d_tr == d_ho))
    exact_copies = float(np.mean(d_tr < 1e-9))
    # record-level attack: score each real record by closeness to the synthetic data
    nn_s = NearestNeighbors(n_neighbors=1).fit(S)
    s_mem = -nn_s.kneighbors(Tr)[0][:, 0]
    s_non = -nn_s.kneighbors(Ho)[0][:, 0]
    def auc(m_mask, n_mask):
        y = np.r_[np.ones(m_mask.sum()), np.zeros(n_mask.sum())]
        return float(roc_auc_score(y, np.r_[s_mem[m_mask], s_non[n_mask]]))
    def tpr_at_fpr(m_mask, n_mask, fpr=0.01):
        thr = np.quantile(s_non[n_mask], 1 - fpr)
        return float(np.mean(s_mem[m_mask] > thr))
    allm, alln = np.ones(len(Tr), bool), np.ones(len(Ho), bool)
    return {
        "dcr_share_train": dcr_share,
        "exact_copy_rate": exact_copies,
        "median_dcr_train": float(np.median(d_tr)),
        "median_dcr_holdout": float(np.median(d_ho)),
        "mia_auc_all": auc(allm, alln),
        "mia_auc_outliers": auc(outlier_mask_members, outlier_mask_nonmembers),
        "mia_auc_typical": auc(~outlier_mask_members, ~outlier_mask_nonmembers),
        "mia_tpr@1%fpr_all": tpr_at_fpr(allm, alln),
        "mia_tpr@1%fpr_outliers": tpr_at_fpr(outlier_mask_members, outlier_mask_nonmembers),
        "mia_tpr@1%fpr_typical": tpr_at_fpr(~outlier_mask_members, ~outlier_mask_nonmembers),
    }


def outlier_masks(train, holdout, ref, enc, k=10):
    """Outlier = top OUTLIER_FRAC by mean distance to k nearest neighbors in an
    independent reference sample of the population (not used for training)."""
    nn = NearestNeighbors(n_neighbors=k).fit(enc.transform(ref))
    score = lambda d: nn.kneighbors(enc.transform(d))[0].mean(axis=1)
    s_tr, s_ho = score(train), score(holdout)
    cut = np.quantile(np.r_[s_tr, s_ho], 1 - OUTLIER_FRAC)
    return s_tr > cut, s_ho > cut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--data", default=None, help="local copy of adult-all.csv")
    ap.add_argument("--models", nargs="+", default=[
        "Copy (reference)", "Interpolation (SMOTE-style)", "Gaussian copula", "CTGAN", "TVAE"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_adult(args.data)
    rows = []
    for seed in args.seeds:
        perm = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        train = perm.iloc[:N_TRAIN].reset_index(drop=True)
        holdout = perm.iloc[N_TRAIN:N_TRAIN + N_HOLDOUT].reset_index(drop=True)
        test = perm.iloc[N_TRAIN + N_HOLDOUT:N_TRAIN + N_HOLDOUT + N_TEST].reset_index(drop=True)
        ref = perm.iloc[N_TRAIN + N_HOLDOUT + N_TEST:].reset_index(drop=True)
        enc = Encoder().fit(train)
        om, on = outlier_masks(train, holdout, ref, enc)
        acc, f1 = tstr(train, test)
        rows.append({"seed": seed, "model": "Real data (TRTR)", "tstr_acc": acc, "tstr_f1": f1})
        for name in args.models:
            t0 = time.time()
            syn = generate(name, train, N_TRAIN, seed)
            acc, f1 = tstr(syn, test)
            r = {"seed": seed, "model": name, "tstr_acc": acc, "tstr_f1": f1,
                 **privacy(syn, train, holdout, enc, om, on),
                 "fit_seconds": time.time() - t0}
            rows.append(r)
            print(json.dumps(r), flush=True)
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)


if __name__ == "__main__":
    main()

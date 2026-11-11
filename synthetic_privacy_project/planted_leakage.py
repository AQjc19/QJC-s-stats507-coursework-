"""Controlled test of H2: can concentrated leakage of outlier records hide behind
an aggregate DCR check that looks normal?

Start from TVAE synthetic data, replace a small fraction of rows with near-copies of
randomly chosen *outlier* training records (numeric columns jittered by 5% of a
standard deviation so they are not exact duplicates), and recompute the checks.

Usage: python planted_leakage.py --out results --seeds 0 1 2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiment import (N_HOLDOUT, N_TEST, N_TRAIN, NUM, Encoder, generate, load_adult,
                        outlier_masks, privacy)

FRACTIONS = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.05]


def plant(syn, train, outlier_idx, frac, rng, enc):
    syn = syn.copy()
    syn[NUM] = syn[NUM].astype(float)
    k = int(round(frac * len(syn)))
    if k == 0:
        return syn
    src = train.iloc[rng.choice(outlier_idx, k, replace=False)].reset_index(drop=True)
    noise = rng.normal(0, 0.05, size=(k, len(NUM))) * enc.sd.to_numpy()
    src[NUM] = src[NUM].to_numpy() + noise
    rows = rng.choice(len(syn), k, replace=False)
    syn.iloc[rows] = src[syn.columns].to_numpy()
    return syn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    df = load_adult(args.data)
    rows = []
    for seed in args.seeds:
        rng = np.random.default_rng(100 + seed)
        perm = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        train = perm.iloc[:N_TRAIN].reset_index(drop=True)
        holdout = perm.iloc[N_TRAIN:N_TRAIN + N_HOLDOUT].reset_index(drop=True)
        ref = perm.iloc[N_TRAIN + N_HOLDOUT + N_TEST:].reset_index(drop=True)
        enc = Encoder().fit(train)
        om, on = outlier_masks(train, holdout, ref, enc)
        base = generate("TVAE", train, N_TRAIN, seed)
        for frac in FRACTIONS:
            syn = plant(base, train, np.flatnonzero(om), frac, rng, enc)
            r = {"seed": seed, "planted_frac": frac, "planted_records": int(round(frac * len(syn))),
                 **privacy(syn, train, holdout, enc, om, on)}
            rows.append(r)
            print(r, flush=True)
    pd.DataFrame(rows).to_csv(Path(args.out) / "planted_leakage.csv", index=False)


if __name__ == "__main__":
    main()

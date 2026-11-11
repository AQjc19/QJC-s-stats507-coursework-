"""Summarize results/results.csv into a table and the progress-report figures."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

res = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
df = pd.read_csv(res / "results.csv")
order = ["Copy (reference)", "Interpolation (SMOTE-style)", "CTGAN", "TVAE", "Gaussian copula"]
order = [m for m in order if m in set(df.model)]
cols = ["tstr_acc", "tstr_f1", "dcr_share_train", "exact_copy_rate", "mia_auc_all",
        "mia_auc_typical", "mia_auc_outliers", "mia_tpr@1%fpr_all",
        "mia_tpr@1%fpr_typical", "mia_tpr@1%fpr_outliers"]
g = df.groupby("model")[cols]
summary = g.mean().join(g.std(), rsuffix="_sd").reindex(["Real data (TRTR)"] + order)
summary.to_csv(res / "summary.csv")
print(summary[cols].round(3).to_string())
print("seeds:", sorted(df.seed.unique()))

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"

# Figure 1: attack success for typical vs outlier records (dumbbell), per generator
fig, ax = plt.subplots(figsize=(6.2, 2.6))
s = summary.loc[order]
y = range(len(order))[::-1]
for yi, m in zip(y, order):
    t, o = s.loc[m, "mia_tpr@1%fpr_typical"], s.loc[m, "mia_tpr@1%fpr_outliers"]
    ax.plot([t, o], [yi, yi], color="#c3c2b7", lw=2, zorder=1)
    ax.errorbar(t, yi, xerr=s.loc[m, "mia_tpr@1%fpr_typical_sd"], fmt="o", color=BLUE,
                ms=7, mec="white", mew=1.5, capsize=2, zorder=3, label="Typical records" if yi == y[0] else None)
    ax.errorbar(o, yi, xerr=s.loc[m, "mia_tpr@1%fpr_outliers_sd"], fmt="o", color=ORANGE,
                ms=7, mec="white", mew=1.5, capsize=2, zorder=3, label="Outlier records (top 10%)" if yi == y[0] else None)
ax.axvline(0.01, color=MUTED, lw=1, ls=":", zorder=0)
ax.text(0.016, y[0] + 0.35, "random guess (1%)", color=MUTED, fontsize=7.5, va="center")
ax.set_yticks(list(y), order)
ax.set_xlabel("Members identified at 1% false-positive rate (TPR)")
ax.set_xlim(0, 0.75)
ax.set_ylim(y[-1] - 0.5, y[0] + 0.7)
ax.grid(axis="x", color="#e5e4e0", lw=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(res / "fig_outlier_risk.pdf")
fig.savefig(res / "fig_outlier_risk.png", dpi=200)

# Figure 2: utility vs the aggregate privacy check
fig, ax = plt.subplots(figsize=(4.4, 2.9))
offsets = {"Copy (reference)": (-8, 8, "right"), "Interpolation (SMOTE-style)": (-8, -14, "right"),
           "CTGAN": (8, 4, "left"), "TVAE": (8, -10, "left"), "Gaussian copula": (8, 6, "left")}
for m in order:
    ax.errorbar(s.loc[m, "dcr_share_train"], s.loc[m, "tstr_f1"],
                xerr=s.loc[m, "dcr_share_train_sd"], yerr=s.loc[m, "tstr_f1_sd"],
                fmt="o", color=BLUE, ms=7, mec="white", mew=1.5, capsize=2)
    dx, dy, ha = offsets.get(m, (6, -10, "left"))
    ax.annotate(m, (s.loc[m, "dcr_share_train"], s.loc[m, "tstr_f1"]), xytext=(dx, dy),
                textcoords="offset points", fontsize=7.5, color=INK, ha=ha)
real_f1 = summary.loc["Real data (TRTR)", "tstr_f1"]
ax.axhline(real_f1, color=MUTED, lw=1, ls=":")
ax.text(0.62, real_f1 + 0.02, "trained on real data", color=MUTED, fontsize=7.5)
ax.axvline(0.5, color=MUTED, lw=1, ls=":")
ax.set_xlabel("Share of synthetic records closer to train than holdout\n(aggregate check; 0.5 = no sign of copying)")
ax.set_ylabel("TSTR F1 on real test data")
ax.set_xlim(0.45, 1.05)
ax.set_ylim(0, 0.8)
fig.tight_layout()
fig.savefig(res / "fig_utility_vs_dcr.pdf")
fig.savefig(res / "fig_utility_vs_dcr.png", dpi=200)

# Figure 3: planted leakage of outlier records (H2)
pl_path = res / "planted_leakage.csv"
if pl_path.exists():
    pl = pd.read_csv(pl_path)
    pg = pl.groupby("planted_records")
    pm, ps = pg.mean(numeric_only=True), pg.std(numeric_only=True)
    x = pm.index.to_numpy()
    clean = df[df.model.isin(["CTGAN", "TVAE", "Gaussian copula"])]["dcr_share_train"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.6), sharex=True)
    a1.axhspan(clean.min(), clean.max(), color="#e5e4e0", lw=0)
    a1.text(x.max(), clean.max() + 0.004, "range for unaltered CTGAN, TVAE,\ncopula (3 seeds)",
            ha="right", va="bottom", fontsize=7, color=MUTED)
    a1.errorbar(x, pm["dcr_share_train"], yerr=ps["dcr_share_train"], color=BLUE, lw=2,
                marker="o", ms=5, mec="white", capsize=2)
    a1.set_ylim(0.44, 0.62)
    a1.set_ylabel("Aggregate DCR share")
    a1.set_title("(a) What the aggregate check sees", fontsize=9, loc="left")
    for col, c, lab in [("mia_tpr@1%fpr_outliers", ORANGE, "Outlier members"),
                        ("mia_tpr@1%fpr_all", BLUE, "All members")]:
        a2.errorbar(x, pm[col], yerr=ps[col], color=c, lw=2, marker="o", ms=5, mec="white",
                    capsize=2, label=lab)
    a2.set_ylabel("Identified at 1% FPR")
    a2.set_ylim(0, 0.6)
    a2.set_title("(b) What an attacker gets", fontsize=9, loc="left")
    a2.legend(frameon=False, fontsize=7.5, loc="upper left")
    for a in (a1, a2):
        a.set_xlabel("Outlier near-copies planted (of 5,000 rows)")
        a.grid(axis="y", color="#e5e4e0", lw=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(res / "fig_planted.pdf")
    fig.savefig(res / "fig_planted.png", dpi=200)
    print(pm[["dcr_share_train", "exact_copy_rate", "mia_tpr@1%fpr_all",
              "mia_tpr@1%fpr_outliers", "mia_tpr@1%fpr_typical"]].round(3).to_string())

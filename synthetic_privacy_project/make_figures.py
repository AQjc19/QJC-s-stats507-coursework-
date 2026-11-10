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
ax.text(0.012, y[-1] - 0.55, "random guess (1%)", color=MUTED, fontsize=7.5, va="bottom")
ax.set_yticks(list(y), order)
ax.set_xlabel("Members identified at 1% false-positive rate (TPR)")
ax.set_xlim(0, 0.75)
ax.grid(axis="x", color="#e5e4e0", lw=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(res / "fig_outlier_risk.pdf")
fig.savefig(res / "fig_outlier_risk.png", dpi=200)

# Figure 2: utility vs the aggregate privacy check
fig, ax = plt.subplots(figsize=(4.4, 2.9))
for m in order:
    ax.errorbar(s.loc[m, "dcr_share_train"], s.loc[m, "tstr_f1"],
                xerr=s.loc[m, "dcr_share_train_sd"], yerr=s.loc[m, "tstr_f1_sd"],
                fmt="o", color=BLUE, ms=7, mec="white", mew=1.5, capsize=2)
    ax.annotate(m, (s.loc[m, "dcr_share_train"], s.loc[m, "tstr_f1"]), xytext=(6, -10),
                textcoords="offset points", fontsize=7.5, color=INK)
real_f1 = summary.loc["Real data (TRTR)", "tstr_f1"]
ax.axhline(real_f1, color=MUTED, lw=1, ls=":")
ax.text(0.52, real_f1 + 0.015, "trained on real data", color=MUTED, fontsize=7.5)
ax.axvline(0.5, color=MUTED, lw=1, ls=":")
ax.set_xlabel("Share of synthetic records closer to train than holdout\n(aggregate check; 0.5 = no sign of copying)")
ax.set_ylabel("TSTR F1 on real test data")
ax.set_xlim(0.45, 1.12)
ax.set_ylim(0, 0.8)
fig.tight_layout()
fig.savefig(res / "fig_utility_vs_dcr.pdf")
fig.savefig(res / "fig_utility_vs_dcr.png", dpi=200)

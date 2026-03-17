"""
01_eda.py  —  Exploratory Data Analysis
Run this first. It shows you exactly what is in the dataset before
any model touches it.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load ──────────────────────────────────────────────────────────
df = pd.read_csv("/mnt/user-data/uploads/dataset.csv")
FEATURES = ["free", "bonus", "investment", "gift"]

print("=" * 55)
print("  RAW DATA — first 10 rows")
print("=" * 55)
print(df.head(10).to_string())

# ── Basic counts ──────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  SHAPE & TYPES")
print("=" * 55)
print(f"  Rows     : {df.shape[0]}")
print(f"  Columns  : {df.shape[1]}  →  {list(df.columns)}")
print(f"  Dtypes   :\n{df.dtypes.to_string()}")

print("\n" + "=" * 55)
print("  CLASS DISTRIBUTION")
print("=" * 55)
counts = df["label"].value_counts()
for label, n in counts.items():
    bar = "█" * n
    print(f"  {label:6}  {n:3}  {bar}  ({n/len(df)*100:.1f}%)")

# ── Per-keyword rates ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  KEYWORD PRESENCE RATE BY CLASS")
print("  (how often each keyword appears in scam vs normal)")
print("=" * 55)
print(f"  {'Keyword':<14} {'In SCAM':>9}  {'In NORMAL':>10}  {'Difference':>11}")
print(f"  {'─'*50}")
for feat in FEATURES:
    r_scam   = df[df["label"] == "scam"][feat].mean() * 100
    r_normal = df[df["label"] == "normal"][feat].mean() * 100
    diff     = r_scam - r_normal
    signal   = "← scam signal" if diff > 10 else ("← normal signal" if diff < -10 else "← weak")
    print(f"  {feat:<14} {r_scam:>8.1f}%  {r_normal:>9.1f}%  {diff:>+10.1f}%  {signal}")

# ── Keyword co-occurrence ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  HOW MANY KEYWORDS PER MESSAGE")
print("=" * 55)
df["kw_count"] = df[FEATURES].sum(axis=1)
for label in ["scam", "normal"]:
    sub = df[df["label"] == label]["kw_count"]
    print(f"\n  {label.upper()}")
    for n in range(5):
        c   = (sub == n).sum()
        bar = "█" * c
        print(f"    {n} keywords : {c:3}  {bar}")

# ── Correlation between features ──────────────────────────────────
print("\n" + "=" * 55)
print("  FEATURE CORRELATION MATRIX")
print("  (do keywords tend to appear together?)")
print("=" * 55)
corr = df[FEATURES].corr().round(2)
print(corr.to_string())
print("\n  bonus & investment correlation:",
      f"{corr.loc['bonus','investment']:.2f}",
      "← these two tend to appear together in scam messages")

# ── Plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("EDA — Scam Message Dataset", fontsize=13, fontweight="bold")

# Class balance
counts.plot(kind="bar", ax=axes[0], color=["#E24B4A","#3FB950"],
            edgecolor="white", rot=0)
axes[0].set_title("Class balance")
axes[0].set_ylabel("Count")
for p in axes[0].patches:
    axes[0].text(p.get_x()+p.get_width()/2, p.get_height()+0.3,
                 str(int(p.get_height())), ha="center", fontsize=11)

# Keyword rate
rates = df.groupby("label")[FEATURES].mean() * 100
x = np.arange(len(FEATURES)); w = 0.35
axes[1].bar(x-w/2, rates.loc["scam"],   width=w, label="Scam",   color="#E24B4A", alpha=0.85)
axes[1].bar(x+w/2, rates.loc["normal"], width=w, label="Normal", color="#3FB950", alpha=0.85)
axes[1].set_xticks(x); axes[1].set_xticklabels([f.upper() for f in FEATURES])
axes[1].set_title("Keyword rate by class"); axes[1].set_ylabel("%")
axes[1].legend(); axes[1].set_ylim(0, 100)

# Keyword count histogram
for label, color in [("scam","#E24B4A"), ("normal","#3FB950")]:
    axes[2].hist(df[df["label"]==label]["kw_count"],
                 bins=[-0.5,0.5,1.5,2.5,3.5,4.5],
                 color=color, alpha=0.75, label=label.title(), edgecolor="white")
axes[2].set_xlabel("Keywords per message"); axes[2].set_ylabel("Count")
axes[2].set_title("Keyword count distribution"); axes[2].legend()

plt.tight_layout()
plt.savefig("eda_output.png", bbox_inches="tight", dpi=150)
plt.close()
print("\n  Plot saved → eda_output.png")
print("\n  WHAT TO TAKE FROM THIS:")
print("  • bonus + investment are near-perfect scam signals")
print("  • free  is noise — appears in both classes equally")
print("  • gift  slightly favours normal messages")
print("  • Most scam messages have 2-3 keywords at once")
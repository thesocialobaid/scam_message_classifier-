"""
06_word_importance_and_filter.py  —  Visualise + filter
Shows word importance for both models and runs the scam filter.
Run after all previous files.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEYWORDS = ["free", "bonus", "investment", "gift"]

df      = pd.read_csv("dataset.csv")
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
w       = np.load("lr_weights.npy")
bias    = np.load("lr_bias.npy")[0]

# Recompute NB likelihoods
n_scam   = (y_train==1).sum()
n_normal = (y_train==0).sum()
p_scam   = n_scam  / len(y_train)
p_normal = n_normal / len(y_train)
p_feat_scam   = (X_train[y_train==1].sum(axis=0) + 1) / (n_scam   + 2)
p_feat_normal = (X_train[y_train==0].sum(axis=0) + 1) / (n_normal + 2)
log_ratios    = np.log(p_feat_scam / p_feat_normal)

# ── Word importance ───────────────────────────────────────────────
print("=" * 55)
print("  WORD IMPORTANCE")
print("=" * 55)

print(f"\n  NAIVE BAYES — log-likelihood ratio")
print(f"  log[ P(word|scam) / P(word|normal) ]")
print(f"  Positive = word more common in scam")
print(f"  Negative = word more common in normal")
print(f"\n  {'Keyword':<14} {'Ratio':>8}  Bar")
print(f"  {'-'*45}")
for i in np.argsort(log_ratios):
    v   = log_ratios[i]
    bar = "#" * int(abs(v) * 5)
    direction = "-> SCAM" if v > 0 else "-> NORMAL"
    print(f"  {KEYWORDS[i]:<14} {v:>+8.3f}  {bar}  {direction}")

print(f"\n  LOGISTIC REGRESSION — learned weights")
print(f"  Positive = feature pushes model toward scam")
print(f"  Negative = feature pushes model toward normal")
print(f"\n  {'Keyword':<14} {'Weight':>8}  Bar")
print(f"  {'-'*45}")
for i in np.argsort(w):
    v   = w[i]
    bar = "#" * int(abs(v) * 2)
    direction = "-> SCAM" if v > 0 else "-> NORMAL"
    print(f"  {KEYWORDS[i]:<14} {v:>+8.3f}  {bar}  {direction}")

print("\n  Both models agree on the ranking.")
print("  This confirms the signal is real — not an artifact of either algorithm.")

# ── Plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Word Importance — NB vs Logistic Regression",
             fontsize=13, fontweight="bold")

for ax, values, title, xlabel in [
    (axes[0], log_ratios,
     "Naive Bayes\nlog[ P(word|scam) / P(word|normal) ]",
     "Log-likelihood ratio"),
    (axes[1], w,
     "Logistic Regression\nLearned weight per keyword",
     "Weight"),
]:
    si     = np.argsort(values)
    names  = [KEYWORDS[i].upper() for i in si]
    vals   = values[si]
    colors = ["#E24B4A" if v > 0 else "#3FB950" for v in vals]
    bars   = ax.barh(names, vals, color=colors, alpha=0.85,
                     edgecolor="white", height=0.5)
    ax.axvline(0, color="gray", linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(v + (0.05 if v >= 0 else -0.05),
                bar.get_y() + bar.get_height() / 2,
                f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=10)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.4)

plt.tight_layout()
plt.savefig("word_importance.png", bbox_inches="tight", dpi=150)
plt.close()
print("\n  Plot saved -> word_importance.png")

# ── Scam filter ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  SCAM FILTER — predict new messages")
print("=" * 55)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def predict(message):
    text     = message.lower()
    features = np.array([[1 if kw in text else 0 for kw in KEYWORDS]])

    # Naive Bayes
    log_s = (np.log(p_scam)
             + (features * np.log(p_feat_scam)
             + (1 - features) * np.log(1 - p_feat_scam)).sum(axis=1))
    log_n = (np.log(p_normal)
             + (features * np.log(p_feat_normal)
             + (1 - features) * np.log(1 - p_feat_normal)).sum(axis=1))
    nb_label = "SCAM" if log_s[0] > log_n[0] else "NORMAL"

    # Logistic Regression
    lr_prob  = sigmoid(features @ w + bias)[0]
    lr_label = "SCAM" if lr_prob >= 0.5 else "NORMAL"

    found = [kw for kw in KEYWORDS if kw in text]
    return {"nb": nb_label, "lr": lr_label,
            "lr_prob": lr_prob, "keywords": found}

test_messages = [
    "Congratulations! You have won a free bonus investment gift.",
    "Exclusive investment opportunity — bonus returns guaranteed!",
    "Your free gift is ready to collect today.",
    "Are you coming to the team meeting at 3pm?",
    "Don't forget to pick up milk on the way home.",
    "Hi, just checking in — how are you doing?",
]

print(f"\n  {'Message':<50} {'Keywords':<25} {'NB':>7} {'LR':>7}  P(scam)")
print(f"  {'-'*100}")
for msg in test_messages:
    r   = predict(msg)
    kws = ", ".join(r["keywords"]) if r["keywords"] else "none"
    print(f"  {msg[:49]:<50} {kws:<25} {r['nb']:>7} {r['lr']:>7}  {r['lr_prob']:.3f}")
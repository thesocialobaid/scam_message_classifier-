"""
03_naive_bayes.py  —  Naive Bayes from scratch
Trains on X_train, evaluates on X_test.
Prints every intermediate value so you can follow the maths.
Run after 02_feature_engineering.py.
"""

import numpy as np

KEYWORDS = ["free", "bonus", "investment", "gift"]

X_train = np.load("X_train.npy")
X_test  = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")

# ── TRAINING ──────────────────────────────────────────────────────
print("=" * 55)
print("  TRAINING — Naive Bayes is just counting")
print("=" * 55)

n_total  = len(y_train)
n_scam   = (y_train == 1).sum()
n_normal = (y_train == 0).sum()

print(f"\n  Training set  : {n_total} messages")
print(f"  Scam messages : {n_scam}")
print(f"  Normal msgs   : {n_normal}")

# Step 1 — priors
p_scam   = n_scam   / n_total
p_normal = n_normal / n_total

print(f"\n  PRIORS (base rate before seeing any keywords):")
print(f"  P(scam)   = {n_scam}/{n_total} = {p_scam:.4f}")
print(f"  P(normal) = {n_normal}/{n_total} = {p_normal:.4f}")

# Step 2 — likelihoods with Laplace smoothing
# For each keyword, count how many scam/normal messages contain it
scam_rows   = X_train[y_train == 1]   # only scam rows
normal_rows = X_train[y_train == 0]   # only normal rows

keyword_counts_scam   = scam_rows.sum(axis=0)    # sum each column
keyword_counts_normal = normal_rows.sum(axis=0)

print(f"\n  RAW KEYWORD COUNTS in training data:")
print(f"  {'Keyword':<14} {'Count in scam':>14} {'Count in normal':>16}")
print(f"  {'─'*46}")
for i, kw in enumerate(KEYWORDS):
    print(f"  {kw:<14} {keyword_counts_scam[i]:>14} {keyword_counts_normal[i]:>16}")

# Apply Laplace smoothing: add 1 to every count, add 2 to denominator
# This prevents any probability from being exactly 0
p_feat_scam   = (keyword_counts_scam   + 1) / (n_scam   + 2)
p_feat_normal = (keyword_counts_normal + 1) / (n_normal + 2)

print(f"\n  LIKELIHOODS P(keyword=1 | class)  [with Laplace smoothing +1]:")
print(f"  {'Keyword':<14} {'P(k|scam)':>10} {'P(k|normal)':>12}  Ratio")
print(f"  {'─'*50}")
for i, kw in enumerate(KEYWORDS):
    ratio = p_feat_scam[i] / p_feat_normal[i]
    print(f"  {kw:<14} {p_feat_scam[i]:>10.4f} {p_feat_normal[i]:>12.4f}  {ratio:.2f}x")

print("\n  ratio > 1 → keyword more common in scam")
print("  ratio < 1 → keyword more common in normal")

# ── PREDICTION ────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  PREDICTION — manual walkthrough on one test message")
print("=" * 55)

# Walk through one example manually
example    = X_test[0]
true_label = "scam" if y_test[0] == 1 else "normal"
print(f"\n  Message features : {dict(zip(KEYWORDS, example))}")
print(f"  True label       : {true_label}")

print(f"\n  Computing log P(scam | message):")
log_scam = np.log(p_scam)
print(f"    start with log P(scam) = log({p_scam:.4f}) = {log_scam:.4f}")
for i, kw in enumerate(KEYWORDS):
    if example[i] == 1:
        contrib = np.log(p_feat_scam[i])
        print(f"    + log P({kw}=1|scam) = log({p_feat_scam[i]:.4f}) = {contrib:.4f}")
        log_scam += contrib
    else:
        contrib = np.log(1 - p_feat_scam[i])
        print(f"    + log P({kw}=0|scam) = log({1-p_feat_scam[i]:.4f}) = {contrib:.4f}")
        log_scam += contrib
print(f"    TOTAL log score (scam)   = {log_scam:.4f}")

print(f"\n  Computing log P(normal | message):")
log_normal = np.log(p_normal)
print(f"    start with log P(normal) = log({p_normal:.4f}) = {log_normal:.4f}")
for i, kw in enumerate(KEYWORDS):
    if example[i] == 1:
        contrib = np.log(p_feat_normal[i])
        print(f"    + log P({kw}=1|normal) = log({p_feat_normal[i]:.4f}) = {contrib:.4f}")
        log_normal += contrib
    else:
        contrib = np.log(1 - p_feat_normal[i])
        print(f"    + log P({kw}=0|normal) = log({1-p_feat_normal[i]:.4f}) = {contrib:.4f}")
        log_normal += contrib
print(f"    TOTAL log score (normal) = {log_normal:.4f}")

winner = "scam" if log_scam > log_normal else "normal"
print(f"\n  {log_scam:.4f} vs {log_normal:.4f}  →  predict: {winner.upper()}")
print(f"  Correct: {'YES' if winner == true_label else 'NO'}")

# ── PREDICT ALL TEST MESSAGES ─────────────────────────────────────
def nb_predict(X):
    log_s = np.log(p_scam)   + (X*np.log(p_feat_scam)   + (1-X)*np.log(1-p_feat_scam)).sum(axis=1)
    log_n = np.log(p_normal) + (X*np.log(p_feat_normal) + (1-X)*np.log(1-p_feat_normal)).sum(axis=1)
    return (log_s > log_n).astype(int)

preds = nb_predict(X_test)

print("\n" + "=" * 55)
print("  ALL TEST PREDICTIONS")
print("=" * 55)
print(f"  {'#':<4} {'Features':<30} {'True':>8} {'Pred':>8}  {'':>4}")
print(f"  {'─'*55}")
for i in range(len(X_test)):
    features = str(dict(zip(KEYWORDS, X_test[i].astype(int))))
    true_lbl = "scam"   if y_test[i] == 1 else "normal"
    pred_lbl = "scam"   if preds[i]  == 1 else "normal"
    correct  = "OK" if true_lbl == pred_lbl else "WRONG"
    print(f"  {i:<4} {features:<30} {true_lbl:>8} {pred_lbl:>8}  {correct}")

# Save predictions for 05_evaluation.py
np.save("nb_preds.npy", preds)
print("\n  NB predictions saved → nb_preds.npy")

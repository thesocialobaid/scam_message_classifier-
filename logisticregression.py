"""
04_logistic_regression.py  —  Logistic Regression from scratch
Gradient descent step by step.
Run after 02_feature_engineering.py.
"""

import numpy as np
 
KEYWORDS = ["free", "bonus", "investment", "gift"]
 
X_train = np.load("X_train.npy")
X_test  = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")
 
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# ── Show what sigmoid does ────────────────────────────────────────
print("=" * 55)
print("  THE SIGMOID FUNCTION")
print("  sigmoid(z) = 1 / (1 + e^-z)")
print("  It converts any number into a probability 0-1")
print("=" * 55)
print(f"\n  {'z':>8}  ->  sigmoid(z)")
print(f"  {'-'*25}")
for z in [-5, -2, -1, 0, 1, 2, 5]:
    bar_pos = int(sigmoid(z) * 20)
    bar = "-" * bar_pos + "o"
    print(f"  {z:>8}  ->  {sigmoid(z):.4f}  {bar}")
print("\n  z < 0  ->  probability < 0.5  ->  predict NORMAL")
print("  z > 0  ->  probability > 0.5  ->  predict SCAM")
print("  z = 0  ->  probability = 0.5  ->  decision boundary")

# ── Training ──────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  TRAINING - gradient descent")
print("  We adjust weights 1000 times, each time reducing error")
print("=" * 55)
 
w  = np.zeros(len(KEYWORDS))
b  = 0.0
lr = 0.1
 
print(f"\n  Starting weights: {w}  bias: {b}")
print(f"  Learning rate   : {lr}")
print(f"\n  {'Iter':>6}  {'Loss':>10}  weights (free, bonus, invest, gift)")
print(f"  {'-'*65}")
 
loss_history = []
PRINT_AT = [0, 1, 5, 10, 50, 100, 500, 999]

for i in range(1000):
    z_val = X_train @ w + b
    p     = sigmoid(z_val)
    err   = p - y_train
 
    loss  = -np.mean(y_train * np.log(p + 1e-15) + (1 - y_train) * np.log(1 - p + 1e-15))
    loss_history.append(loss)
 
    grad_w = (X_train.T @ err) / len(X_train)
    grad_b = err.mean()
 
    w -= lr * grad_w
    b -= lr * grad_b
 
    if i in PRINT_AT:
        print(f"  {i:>6}  {loss:>10.4f}  {w.round(3)}")
 
print(f"\n  Final weights:")
print(f"  {'Keyword':<14} {'Weight':>8}  Meaning")
print(f"  {'-'*45}")
for kw, wi in zip(KEYWORDS, w):
    direction = "-> scam signal" if wi > 0.5 else ("-> normal signal" if wi < -0.1 else "-> weak")
    print(f"  {kw:<14} {wi:>8.3f}  {direction}")
print(f"  {'bias':<14} {b:>8.3f}")
 
# ── What the weights mean ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  WHAT THE WEIGHTS MEAN")
print("  For a message with bonus=1, investment=1, rest=0:")
print("=" * 55)
 
example_feat = np.array([0, 1, 1, 0])
z_example    = example_feat @ w + b
p_example    = sigmoid(z_example)
 
print(f"\n  z = 0*{w[0]:.3f} + 1*{w[1]:.3f} + 1*{w[2]:.3f} + 0*{w[3]:.3f} + ({b:.3f})")
print(f"    = {z_example:.4f}")
print(f"  P(scam) = sigmoid({z_example:.4f}) = {p_example:.4f}")
print(f"  Prediction: {'SCAM' if p_example >= 0.5 else 'NORMAL'}")
 
# ── Predict all test messages ─────────────────────────────────────
preds = (sigmoid(X_test @ w + b) >= 0.5).astype(int)
proba = sigmoid(X_test @ w + b)
 
print("\n" + "=" * 55)
print("  ALL TEST PREDICTIONS")
print("=" * 55)
print(f"  {'#':<4} {'free':>5} {'bonus':>6} {'invest':>8} {'gift':>5} {'P(scam)':>8} {'True':>8} {'Pred':>8} {'':>5}")
print(f"  {'-'*62}")
for i in range(len(X_test)):
    f = X_test[i]
    true_lbl = "scam"   if y_test[i] == 1 else "normal"
    pred_lbl = "scam"   if preds[i]  == 1 else "normal"
    correct  = "OK" if true_lbl == pred_lbl else "WRONG"
    print(f"  {i:<4} {f[0]:>5} {f[1]:>6} {f[2]:>8} {f[3]:>5} {proba[i]:>8.3f} {true_lbl:>8} {pred_lbl:>8}  {correct}")
 
np.save("lr_preds.npy",   preds)
np.save("lr_proba.npy",   proba)
np.save("lr_weights.npy", w)
np.save("lr_bias.npy",    np.array([b]))
print("\n  Saved: lr_preds.npy  lr_proba.npy  lr_weights.npy  lr_bias.npy")
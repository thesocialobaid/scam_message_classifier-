"""
02_feature_engineering.py  —  Text → Numbers
This is the step that converts raw text into something a model
can work with. Run this second.
"""

import numpy as np
import pandas as pd

df = pd.read_csv(r"F:\0- FAST\4- Semester 04\PROBABILITY AND STATS\Assignment 02\Code\dataset.csv")
KEYWORDS = ["free", "bonus", "investment", "gift"]

print("=" * 55)
print("  STEP 1 — what the raw data looks like")
print("=" * 55)
print(df[KEYWORDS + ["label"]].head(8).to_string())

# ── Feature matrix ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 2 — extract feature matrix X and label vector y")
print("  X: each row = one message, each column = one keyword")
print("  y: 1 = scam,  0 = normal")
print("=" * 55)

X = df[KEYWORDS].values               # shape (50, 4) — already binary
y = (df["label"] == "scam").astype(int).values

print(f"\n  X shape : {X.shape}   (50 messages × 4 features)")
print(f"  y shape : {y.shape}   (50 labels)")
print(f"\n  First 8 rows of X:")
print(f"  {'free':>6} {'bonus':>6} {'invest':>8} {'gift':>6}  →  label")
print(f"  {'─'*40}")
for i in range(8):
    row = X[i]
    lbl = "scam" if y[i] == 1 else "normal"
    print(f"  {row[0]:>6} {row[1]:>6} {row[2]:>8} {row[3]:>6}     {lbl}")

# ── Show the function that does this on raw text ──────────────────
print("\n" + "=" * 55)
print("  STEP 3 — the feature engineering function")
print("  (what happens when you get a brand-new raw message)")
print("=" * 55)

def extract_features(message: str, keywords: list) -> np.ndarray:
    """
    Takes a raw string, returns a binary numpy array.
    1 if keyword is in the message, 0 if not.
    This is feature engineering.
    """
    text = message.lower()
    return np.array([1 if kw in text else 0 for kw in keywords])

# Show it working on real examples
test_messages = [
    "Congratulations! Claim your free bonus now.",
    "Are you joining us for dinner tonight?",
    "Exclusive investment opportunity — massive bonus guaranteed.",
    "Your gift is waiting, please collect it.",
]

print(f"\n  {'Message (truncated)':<45} {'free':>5} {'bonus':>6} {'invest':>8} {'gift':>5}")
print(f"  {'─'*72}")
for msg in test_messages:
    feats = extract_features(msg, KEYWORDS)
    print(f"  {msg[:44]:<45} {feats[0]:>5} {feats[1]:>6} {feats[2]:>8} {feats[3]:>5}")

# ── Train/test split ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("  STEP 4 — train / test split")
print("  75% trains the model, 25% evaluates it honestly")
print("=" * 55)

np.random.seed(42)
idx   = np.random.permutation(len(X))
split = int(0.75 * len(X))

X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

print(f"\n  Total  : {len(X)} messages")
print(f"  Train  : {len(X_train)} messages  ({len(X_train)/len(X)*100:.0f}%)")
print(f"  Test   : {len(X_test)} messages  ({len(X_test)/len(X)*100:.0f}%)")
print(f"\n  Class ratio check (should be similar in both splits):")
print(f"  Train scam ratio : {y_train.mean():.2f}")
print(f"  Test  scam ratio : {y_test.mean():.2f}")

print("\n  WHAT JUST HAPPENED:")
print("  Raw text string  →  [1, 1, 0, 0]  →  X matrix  →  ready for model")
print("  This conversion is the entire concept of feature engineering.")
print("  The model never sees the words — only these 0s and 1s.")

# Save so later files can import
np.save("X_train.npy", X_train)
np.save("X_test.npy",  X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy",  y_test)
print("\n  Data saved → /tmp/  (used by files 03, 04, 05)")
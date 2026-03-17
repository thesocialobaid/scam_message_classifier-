"""
05_evaluation.py  —  Metrics from scratch
Shows what every number in the evaluation actually means.
Run after 03_naive_bayes.py and 04_logistic_regression.py.
"""

import numpy as np

KEYWORDS = ["free", "bonus", "investment", "gift"]

y_test   = np.load("y_test.npy")
nb_preds = np.load("nb_preds.npy")
lr_preds = np.load("lr_preds.npy")
lr_proba = np.load("lr_proba.npy")

def full_evaluation(y_true, y_pred, name):
    TP = int(((y_pred==1) & (y_true==1)).sum())
    TN = int(((y_pred==0) & (y_true==0)).sum())
    FP = int(((y_pred==1) & (y_true==0)).sum())
    FN = int(((y_pred==0) & (y_true==1)).sum())

    acc  = (TP+TN) / len(y_true)
    prec = TP/(TP+FP) if TP+FP>0 else 0
    rec  = TP/(TP+FN) if TP+FN>0 else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0

    print(f"\n  {name}")
    print(f"  {'-'*45}")
    print(f"\n  Confusion matrix:")
    print(f"                    Predicted")
    print(f"                    Normal    Scam")
    print(f"  Actual Normal       {TN:>4}    {FP:>4}   (TN | FP)")
    print(f"  Actual Scam         {FN:>4}    {TP:>4}   (FN | TP)")
    print(f"\n  What each cell means:")
    print(f"  TN={TN}  -> normal messages correctly left alone")
    print(f"  FP={FP}  -> normal messages wrongly flagged as scam  (false alarm)")
    print(f"  FN={FN}  -> real scams the model missed              (worst outcome)")
    print(f"  TP={TP}  -> real scams correctly caught")
    print(f"\n  Metrics:")
    print(f"  Accuracy  = (TP+TN)/total = ({TP}+{TN})/{len(y_true)} = {acc:.4f}")
    print(f"  Precision = TP/(TP+FP)   = {TP}/({TP}+{FP})  = {prec:.4f}")
    print(f"             -> of all scam flags, {prec*100:.1f}% were real scams")
    print(f"  Recall    = TP/(TP+FN)   = {TP}/({TP}+{FN})  = {rec:.4f}")
    print(f"             -> of all real scams, {rec*100:.1f}% were caught")
    print(f"  F1        = 2*P*R/(P+R)  = {f1:.4f}")

    return dict(acc=acc, prec=prec, rec=rec, f1=f1,
                TP=TP, TN=TN, FP=FP, FN=FN)

print("=" * 55)
print("  EVALUATION METRICS — both models")
print("=" * 55)

nb_m = full_evaluation(y_test, nb_preds, "NAIVE BAYES")
lr_m = full_evaluation(y_test, lr_preds, "LOGISTIC REGRESSION")

print("\n" + "=" * 55)
print("  SIDE-BY-SIDE COMPARISON")
print("=" * 55)
print(f"\n  {'Metric':<12} {'Naive Bayes':>14} {'Logistic Reg':>14}")
print(f"  {'-'*42}")
for key, label in [("acc","Accuracy"),("prec","Precision"),("rec","Recall"),("f1","F1")]:
    nb_val = nb_m[key]
    lr_val = lr_m[key]
    winner = "<- NB wins" if nb_val > lr_val else ("<- LR wins" if lr_val > nb_val else "<- tie")
    print(f"  {label:<12} {nb_val:>14.4f} {lr_val:>14.4f}  {winner}")

print("\n" + "=" * 55)
print("  WHY RECALL MATTERS MOST FOR A SCAM FILTER")
print("=" * 55)
print("""
  FP (false alarm) -> normal message flagged as scam
  User is mildly annoyed. Not a disaster.

  FN (missed scam) -> real scam delivered to user
  User potentially loses money. This is the bad outcome.

  Therefore: maximise RECALL over precision.
  A model that catches more scams with some false alarms
  is more useful than one that is precise but misses scams.
""")
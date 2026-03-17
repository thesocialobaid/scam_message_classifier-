Scam Message Classifier
Binary text classification built from scratch using NumPy and Pandas — no sklearn models used. Implements Naive Bayes and Logistic Regression from their mathematical foundations to classify messages as scam or normal based on keyword presence.

What this project does
Given a message, the classifier scans for four keywords — free, bonus, investment, gift — and predicts whether the message is a scam. Both models are trained entirely from scratch: Naive Bayes through probability counting, Logistic Regression through gradient descent.

"Exclusive investment opportunity — bonus returns guaranteed!"
→  NB: SCAM (99.7%)   LR: SCAM (98.0%)

"Are you coming to the meeting tomorrow?"
→  NB: NORMAL         LR: NORMAL (20.7%)
Project structure
├── dataset.csv                       raw dataset — 50 messages, 4 keyword features
├── 01_eda.py                         exploratory data analysis + visualisation
├── 02_feature_engineering.py         text → binary feature matrix
├── 03_naive_bayes.py                 Naive Bayes trained from scratch
├── 04_logistic_regression.py         Logistic Regression via gradient descent
├── 05_evaluation.py                  confusion matrix + metrics from scratch
├── 06_word_importance_and_filter.py  word importance charts + scam filter
Dataset
Property	Value
Total messages	50
Scam messages	31 (62%)
Normal messages	19 (38%)
Features	free, bonus, investment, gift
Feature encoding	Binary — 1 if keyword present, 0 if absent
Key finding from EDA: bonus and investment appear in 77% and 74% of scam messages respectively, and in 0% of normal messages — making them near-perfect discriminators. free appears with similar frequency in both classes (52% vs 42%) and carries almost no signal. gift inversely correlates with scam — it is more common in normal messages.

How to run
Clone the repo and place dataset.csv in the same folder as the scripts. Run files in order:

python 01_eda.py
python 02_feature_engineering.py
python 03_naive_bayes.py
python 04_logistic_regression.py
python 05_evaluation.py
python 06_word_importance_and_filter.py
Each file is self-contained and prints its intermediate values to the terminal so every step can be followed line by line.

Requirements

pip install numpy pandas matplotlib
No sklearn, no scipy, no external ML libraries.

Implementation
Feature engineering
Raw messages are converted to a binary vector by scanning for each keyword:

def extract_features(message, keywords):
    text = message.lower()
    return np.array([1 if kw in text else 0 for kw in keywords])

# "Claim your free bonus now."  →  [1, 1, 0, 0]
# "Are you joining us tonight?" →  [0, 0, 0, 0]
Naive Bayes
Training is pure counting. Laplace smoothing (α = 1) prevents zero probabilities. Prediction uses log-probabilities to avoid numerical underflow.

# Training
p_scam         = n_scam / n_total
p_feat_scam    = (X_train[y_train==1].sum(axis=0) + 1) / (n_scam   + 2)
p_feat_normal  = (X_train[y_train==0].sum(axis=0) + 1) / (n_normal + 2)

# Prediction
log_score_scam   = log(p_scam)   + sum(x*log(p_feat_scam)   + (1-x)*log(1-p_feat_scam))
log_score_normal = log(p_normal) + sum(x*log(p_feat_normal) + (1-x)*log(1-p_feat_normal))
predict = (log_score_scam > log_score_normal)
Logistic Regression
Trained via gradient descent over 1000 iterations minimising Binary Cross-Entropy loss.

# Forward pass
z = X_train @ w + b
p = 1 / (1 + exp(-z))

# Gradients
grad_w = (X_train.T @ (p - y_train)) / n
grad_b = (p - y_train).mean()

# Update
w -= 0.1 * grad_w
b -= 0.1 * grad_b
Evaluation metrics
All metrics computed from scratch — no sklearn.

TP = ((y_pred==1) & (y_true==1)).sum()   # correct scam flags
TN = ((y_pred==0) & (y_true==0)).sum()   # correct normal passes
FP = ((y_pred==1) & (y_true==0)).sum()   # false alarms
FN = ((y_pred==0) & (y_true==1)).sum()   # missed scams

precision = TP / (TP + FP)
recall    = TP / (TP + FN)
f1        = 2 * precision * recall / (precision + recall)
Results
Model	Accuracy	Precision	Recall	F1
Naive Bayes	84.6%	100%	71.4%	83.3%
Logistic Regression	69.2%	71.4%	71.4%	71.4%
Naive Bayes produced zero false positives — every message it flagged was a genuine scam. Both models missed the same two scam messages, both of which contained none of the four keywords — a ceiling imposed by the feature set, not the algorithm.

For a scam filter, recall matters more than precision — a missed scam is a worse outcome than a false alarm.

Word importance
Both models independently agree on the same keyword ranking:

Keyword	NB log-ratio	LR weight	Signal
bonus	+2.45	+2.82	Strong scam
investment	+2.39	+2.41	Strong scam
free	+0.55	+1.59	Weak / noisy
gift	−0.55	−0.49	Mild normal
Two completely different mathematical approaches arriving at the same ranking confirms that these patterns are real in the data and not an artifact of either algorithm.

Logistic Regression weight progression
The weights grow through gradient descent as the model discovers each keyword's signal:

Iteration	Loss	free	bonus	investment	gift
0	0.693	0.014	0.026	0.024	0.000
10	0.578	0.118	0.240	0.226	−0.018
100	0.372	0.428	1.091	1.007	−0.369
999	0.252	1.592	2.818	2.407	−0.487
Key concepts demonstrated
Concept	Where it appears
Binary classification	Both models predict scam or normal
Feature engineering	02_feature_engineering.py — text → number matrix
Bayes' Theorem	03_naive_bayes.py — prior × likelihood
Laplace smoothing	03_naive_bayes.py — prevents zero probability
Log-probabilities	03_naive_bayes.py — prevents numerical underflow
Sigmoid function	04_logistic_regression.py — squashes z to probability
Binary cross-entropy	04_logistic_regression.py — loss function
Gradient descent	04_logistic_regression.py — 1000-step weight update loop
Confusion matrix	05_evaluation.py — TP / TN / FP / FN from scratch
Precision vs recall	05_evaluation.py — why recall matters more for filters
Word importance	06_word_importance_and_filter.py — NB ratio vs LR weights
Why from scratch
Using NumPy directly instead of sklearn means every operation is visible. The Naive Bayes training loop is literally counting rows. The Logistic Regression training loop is four lines. There are no black boxes — every number in the output can be traced back to a specific line of arithmetic.

The same models in sklearn take three lines to call. This implementation takes sixty lines to explain. Once you have written the sixty lines, the three-line version is no longer a black box — it is just a shortcut to something you already understand.

Assignment context
Built for Probability and Statistics — Semester 4, FAST University.

Requirements covered:

Build a basic spam/scam filter
Practice feature engineering with text
Train models from scratch
Visualise word importance in classification

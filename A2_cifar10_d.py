"""
@author: Daniel Wolff

"""

# Disclaimer: A LLM was used for enhancing visualisation of results.
# Not run yet, but the code is structured to perform the following:
# - Load CIFAR-10 data
# - Perform accuracy-scored CV using LogisticRegressionCV to find the best C
# - Perform log-loss CV using cross_val_predict to find the best C
# - Plot both CV results on log-log plots for comparison
# - Finally, compare the test set performance of the best models from both CV methods

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import warnings

# Data loading
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='latin1')

batches     = [unpickle(f"./data/cifar-10-batches-py/data_batch_{i}") for i in range(1, 6)]
test        = unpickle("./data/cifar-10-batches-py/test_batch")
label_names = unpickle("./data/cifar-10-batches-py/batches.meta")["label_names"]

X_train = np.concatenate([b["data"]   for b in batches]) / 255.0
y_train = np.concatenate([b["labels"] for b in batches])
X_test  = test["data"] / 255.0
y_test  = np.array(test["labels"])


# (c) Accuracy CV — reuse existing LogisticRegressionCV result
Cs = np.logspace(-4, 1, 10)

print("Running accuracy-scored CV (c)...")
acc_cv_model = LogisticRegressionCV(
    Cs=Cs, cv=4, solver='saga', max_iter=2000,
    scoring='accuracy', refit=True, n_jobs=-1, random_state=42
)
with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    acc_cv_model.fit(X_train, y_train)

# Mean accuracy per C: scores_ shape is {class: (n_folds, n_Cs)}
acc_mean_scores = np.mean(
    [scores.mean(axis=0) for scores in acc_cv_model.scores_.values()], axis=0
)
best_C_acc = acc_cv_model.C_[0]


# ── (d) Log-loss CV using cross_val_predict ────────────────────────────────────
# This is the logarithmic scoring rule: it penalises confident wrong predictions
# much more harshly than accuracy does.

print("Running log-loss CV (d)...")
logloss_scores = []

for C in Cs:
    model = LogisticRegression(
        C=C, solver='saga', max_iter=2000, random_state=42
    )
    # shape: (n_train_samples, n_classes)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        y_prob_oof = cross_val_predict(
            model, X_train, y_train,
            cv=4, method='predict_proba', n_jobs=-1
        )

    # log_loss: lower is better, so we negate it to make higher = better for comparison
    loss = metrics.log_loss(y_train, y_prob_oof)
    logloss_scores.append(loss)
    print(f"  C={C:.6f} | Log-loss={loss:.4f}")

logloss_scores = np.array(logloss_scores)

# Best C is where log-loss is MINIMISED
best_C_logloss = Cs[np.argmin(logloss_scores)]
print(f"\nBest C by accuracy  (c): {best_C_acc:.6f}")
print(f"Best C by log-loss  (d): {best_C_logloss:.6f}")
print(f"Do they agree? {'✅ Yes' if np.isclose(best_C_acc, best_C_logloss, rtol=0.5) else '⚠️  No — different optimal C'}")


# ── Log-log plots ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("CV Scoring Rules Comparison: Accuracy (c) vs Log-Loss (d)",
             fontsize=13, fontweight='bold')

# (c) Accuracy — log-log
ax1.loglog(Cs, acc_mean_scores, marker='o', color='steelblue', linewidth=2)
ax1.axvline(best_C_acc, color='green', linestyle='--', linewidth=2,
            label=f"Best C = {best_C_acc:.4f}")
ax1.scatter([best_C_acc], [acc_mean_scores[np.argmin(np.abs(Cs - best_C_acc))]],
            color='green', zorder=5, s=100)
ax1.set_xlabel("C (log scale)")
ax1.set_ylabel("Mean CV Accuracy (log scale)")
ax1.set_title("(c) Accuracy Scoring")
ax1.legend()
ax1.grid(True, which="both", ls="--", alpha=0.5)

# (d) Log-loss — log-log (lower is better, so we note that on the axis)
ax2.loglog(Cs, logloss_scores, marker='o', color='tomato', linewidth=2)
ax2.axvline(best_C_logloss, color='green', linestyle='--', linewidth=2,
            label=f"Best C = {best_C_logloss:.4f}")
ax2.scatter([best_C_logloss], [logloss_scores[np.argmin(logloss_scores)]],
            color='green', zorder=5, s=100)
ax2.set_xlabel("C (log scale)")
ax2.set_ylabel("Mean CV Log-Loss (log scale) ↓ lower is better")
ax2.set_title("(d) Logarithmic Scoring Rule (Log-Loss)")
ax2.legend()
ax2.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.show()


# ── Final comparison on test set ───────────────────────────────────────────────
# Refit a model with the best log-loss C and evaluate
best_logloss_model = LogisticRegression(
    C=best_C_logloss, solver='saga', max_iter=2000, random_state=42
)
with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    best_logloss_model.fit(X_train, y_train)

y_pred_ll = best_logloss_model.predict(X_test)
y_prob_ll = best_logloss_model.predict_proba(X_test)

print("\n" + "="*55)
print("         FINAL TEST SET COMPARISON")
print("="*55)
print(f"  (c) Best C by accuracy : {best_C_acc:.6f}")
print(f"      Test accuracy      : {metrics.accuracy_score(y_test, acc_cv_model.predict(X_test))*100:.2f}%")
print(f"      Test log-loss      : {metrics.log_loss(y_test, acc_cv_model.predict_proba(X_test)):.4f}")
print()
print(f"  (d) Best C by log-loss : {best_C_logloss:.6f}")
print(f"      Test accuracy      : {metrics.accuracy_score(y_test, y_pred_ll)*100:.2f}%")
print(f"      Test log-loss      : {metrics.log_loss(y_test, y_prob_ll):.4f}")
print("="*55)
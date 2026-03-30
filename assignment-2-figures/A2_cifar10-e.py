import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
import warnings

# ── Data loading ───────────────────────────────────────────────────────────────
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


# ── Fit final model ────────────────────────────────────────────────────────────
C_optimal = 0.016681

print(f"Fitting final model with C = {C_optimal} on full training set...")
model = LogisticRegression(C=C_optimal, solver='saga', max_iter=2000, random_state=42)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    model.fit(X_train, y_train)
    converged = not any("ConvergenceWarning" in str(x.category) for x in w)

print(f"{'✅ Converged' if converged else '⚠️  Did not converge'}")


# ── Predictions & scores ───────────────────────────────────────────────────────
y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)
y_train_prob = model.predict_proba(X_train)
y_test_prob  = model.predict_proba(X_test)

train_acc  = accuracy_score(y_train, y_train_pred)
test_acc   = accuracy_score(y_test,  y_test_pred)
train_loss = log_loss(y_train, y_train_prob)
test_loss  = log_loss(y_test,  y_test_prob)
train_err  = 1 - train_acc
test_err   = 1 - test_acc


# ── Results summary ────────────────────────────────────────────────────────────
print(f"\n{'='*58}")
print(f"  PART (e) — Final Model Results  (C = {C_optimal})")
print(f"{'='*58}")
print(f"  {'Metric':<30} {'Train':>10} {'Test':>10}")
print(f"  {'-'*50}")
print(f"  {'Accuracy':<30} {train_acc:>10.4f} {test_acc:>10.4f}")
print(f"  {'Error (1 - accuracy)':<30} {train_err:>10.4f} {test_err:>10.4f}")
print(f"  {'Log-loss':<30} {train_loss:>10.4f} {test_loss:>10.4f}")
print(f"{'='*58}")
print(f"\n  ➤ Generalisation gap (accuracy): {abs(train_acc - test_acc):.4f}")
print(f"  ➤ Generalisation gap (log-loss): {abs(train_loss - test_loss):.4f}")

print("\nPer-class report on test set:")
print(classification_report(y_test, y_test_pred, target_names=label_names))



# ── Per-class accuracy bar chart ───────────────────────────────────────────────
class_acc = [accuracy_score(y_test[y_test == i], y_test_pred[y_test == i]) for i in range(10)]
colors    = ['steelblue' if a >= test_acc else 'tomato' for a in class_acc]

plt.figure(figsize=(10, 5))
plt.bar(label_names, class_acc, color=colors)
plt.axhline(test_acc, color='green', linestyle='--', linewidth=2, label=f'Overall test acc = {test_acc:.4f}')
plt.xlabel("Class")
plt.ylabel("Accuracy")
plt.title(f"Part (e) — Per-Class Test Accuracy  (C = {C_optimal})")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(True, axis='y', ls="--", alpha=0.5)
plt.tight_layout()
plt.show()
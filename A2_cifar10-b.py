"""
@author: Daniel Wolff

"""

# Disclaimer: A LLM was used for enhancing visualisation of results.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import warnings


# Data loading
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='latin1')

batches    = [unpickle(f"./data/cifar-10-batches-py/data_batch_{i}") for i in range(1, 6)]
test       = unpickle("./data/cifar-10-batches-py/test_batch")
label_names = unpickle("./data/cifar-10-batches-py/batches.meta")["label_names"]

X_train = np.concatenate([b["data"]   for b in batches]) / 255.0
y_train = np.concatenate([b["labels"] for b in batches])
X_test  = test["data"] / 255.0
y_test  = np.array(test["labels"])

def data_to_image(x):
    return x.reshape(3, 32, 32).transpose(1, 2, 0)

def plot_image(image, title=""):
    fig = plt.imshow(data_to_image(image))
    plt.title(title)
    fig.axes.set_axis_off()
    plt.show()


# Training loop
C_values = np.logspace(-3, 0, 8)

results = []

for C in C_values:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        start = time.time()
        model = LogisticRegression(C=C, max_iter=500, solver='lbfgs')
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        converged = not any("ConvergenceWarning" in str(warning.category) for warning in w)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    acc    = accuracy_score(y_test, y_pred)

    results.append({
        "C":          C,
        "accuracy":   acc,
        "time":       elapsed,
        "converged":  converged,
        "model":      model,
        "y_pred":     y_pred,
        "y_prob":     y_prob,
    })

    status = "✅ converged" if converged else "⚠️  did not converge"
    print(f"C={C:.6f} | Acc={acc:.4f} | Time={elapsed:.1f}s | {status}")


# Find best converged result
converged_results = [r for r in results if r["converged"]]
best = max(converged_results, key=lambda r: r["accuracy"]) if converged_results else max(results, key=lambda r: r["accuracy"])

print(f"\n{'='*55}")
print(f"  Best C (converged): {best['C']:.6f}")
print(f"  Test Accuracy     : {best['accuracy']*100:.2f}%")
print(f"  Training Time     : {best['time']:.1f}s")
print(f"{'='*55}\n")
print("Per-class breakdown:")
print(classification_report(y_test, best["y_pred"], target_names=label_names))


# Figure 1: Accuracy & Training Time vs C 
C_arr   = np.array([r["C"]        for r in results])
acc_arr = np.array([r["accuracy"] for r in results])
t_arr   = np.array([r["time"]     for r in results])
conv    = np.array([r["converged"]for r in results])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Logistic Regression on CIFAR-10 — Effect of Regularization", fontsize=14, fontweight='bold')

# Accuracy plot
ax1.semilogx(C_arr[conv],  acc_arr[conv],  'o-', color='steelblue',  label='Converged',     linewidth=2)
ax1.semilogx(C_arr[~conv], acc_arr[~conv], 's--',color='tomato',     label='Not converged', linewidth=1.5, alpha=0.7)
ax1.axvline(best["C"], color='green', linestyle=':', linewidth=2, label=f"Best C={best['C']:.4f}")
ax1.scatter([best["C"]], [best["accuracy"]], color='green', zorder=5, s=100)
ax1.set_xlabel("C (log scale)")
ax1.set_ylabel("Test Accuracy")
ax1.set_title("Test Accuracy vs C")
ax1.legend()
ax1.grid(True, which="both", ls="--", alpha=0.5)

# Training time plot
ax2.semilogx(C_arr[conv],  t_arr[conv],  'o-', color='steelblue',  label='Converged',     linewidth=2)
ax2.semilogx(C_arr[~conv], t_arr[~conv], 's--',color='tomato',     label='Not converged', linewidth=1.5, alpha=0.7)
ax2.axvline(best["C"], color='green', linestyle=':', linewidth=2, label=f"Best C={best['C']:.4f}")
ax2.set_xlabel("C (log scale)")
ax2.set_ylabel("Training Time (seconds)")
ax2.set_title("Training Time vs C")
ax2.legend()
ax2.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.show()


# Figure 2: Confidence distribution for best model
max_probs  = np.max(best["y_prob"], axis=1)
correct    = best["y_pred"] == y_test

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Confidence Analysis", fontsize=13, fontweight='bold')

# Histogram of confidence for correct vs incorrect
axes[0].hist(max_probs[correct],  bins=30, alpha=0.6, color='steelblue', label='Correct')
axes[0].hist(max_probs[~correct], bins=30, alpha=0.6, color='tomato',    label='Incorrect')
axes[0].set_xlabel("Max Predicted Probability")
axes[0].set_ylabel("Count")
axes[0].set_title("Prediction Confidence: Correct vs Incorrect")
axes[0].legend()
axes[0].grid(True, ls="--", alpha=0.5)

# Per-class accuracy bar chart
class_acc = [accuracy_score(y_test[y_test == i], best["y_pred"][y_test == i]) for i in range(10)]
colors = ['steelblue' if a >= best["accuracy"] else 'tomato' for a in class_acc]
axes[1].bar(label_names, class_acc, color=colors)
axes[1].axhline(best["accuracy"], color='green', linestyle='--', label=f"Overall Acc={best['accuracy']:.3f}")
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Per-Class Accuracy")
axes[1].set_xticklabels(label_names, rotation=45, ha='right')
axes[1].legend()
axes[1].grid(True, axis='y', ls="--", alpha=0.5)

plt.tight_layout()
plt.show()
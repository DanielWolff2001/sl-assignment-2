"""
@author: Daniel Wolff 5013542

"""

# Disclaimer: A LLM was used for enhancing visualisation of results.


# The visualisation of CV results was improved with:
# - Plotting mean CV accuracy vs C with a vertical line at the best C
# - A summary of CV results including best C, what it means, mean CV accuracy, and test set accuracy
# - A classification report showing per-class precision, recall, and F1 scores on the test set 

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import warnings

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

dict1 = unpickle("./data/cifar-10-batches-py/data_batch_1")
dict2 = unpickle("./data/cifar-10-batches-py/data_batch_2")
dict3 = unpickle("./data/cifar-10-batches-py/data_batch_3")
dict4 = unpickle("./data/cifar-10-batches-py/data_batch_4")
dict5 = unpickle("./data/cifar-10-batches-py/data_batch_5")
test  = unpickle("./data/cifar-10-batches-py/test_batch")
meta_data  = unpickle("./data/cifar-10-batches-py/batches.meta")
label_names = meta_data["label_names"]

X_train = np.concatenate([d["data"] for d in [dict1,dict2,dict3,dict4,dict5]]) / 255.0
y_train = np.concatenate([d["labels"] for d in [dict1,dict2,dict3,dict4,dict5]])
X_test  = test["data"] / 255.0
y_test  = np.array(test["labels"])

# Log scale from 1e-4 to 1e1 — 10 candidates
Cs = np.logspace(-4, 1, 10)


cv_model = LogisticRegressionCV(
    Cs=Cs,
    cv=4,
    solver='saga',        
    max_iter=2000,        
    refit=True,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("Starting 4-Fold Cross-Validation...")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    cv_model.fit(X_train, y_train)
    non_converged = [str(warning.message) for warning in w if "ConvergenceWarning" in str(warning.category)]

if non_converged:
    print(f"\n⚠️  Convergence warnings for {len(non_converged)} fits — consider increasing max_iter further.")
else:
    print("\n✅ All fits converged successfully.")

# Extract CV scores
mean_scores = np.mean(
    [scores.mean(axis=0) for scores in cv_model.scores_.values()],
    axis=0
)

best_C       = cv_model.C_[0]  # best C found by CV
best_cv_acc  = mean_scores[np.argmin(np.abs(Cs - best_C))]

# Plot mean CV accuracy vs C
plt.figure(figsize=(10, 6))
plt.semilogx(Cs, mean_scores, marker='o', linewidth=2, label='Mean CV Accuracy')
plt.axvline(best_C, color='red', linestyle='--', label=f'Best C = {best_C:.4f}')
plt.scatter([best_C], [best_cv_acc], color='red', zorder=5)
plt.xlabel('Regularization Parameter C (log scale)')
plt.ylabel('Mean CV Accuracy')
plt.title('4-Fold Cross-Validation Accuracy vs C')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Results summary
y_pred      = cv_model.predict(X_test)
test_acc    = accuracy_score(y_test, y_pred)

print("\n" + "="*55)
print("         CROSS-VALIDATION RESULTS SUMMARY")
print("="*55)
print(f"  Best C found        : {best_C:.6f}")
print(f"  What C means        : {'high regularization (simpler model)' if best_C < 0.1 else 'low regularization (complex model)'}")
print(f"  Mean CV Accuracy    : {best_cv_acc:.4f}  ({best_cv_acc*100:.2f}%)")
print(f"  Test Set Accuracy   : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print("="*55)

print("\nPer-class breakdown on test set:")
print(classification_report(y_test, y_pred, target_names=label_names))
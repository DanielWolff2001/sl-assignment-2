from sklearn.linear_model import LogisticRegressionCV
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time


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
test = unpickle("./data/cifar-10-batches-py/test_batch")
meta_data = unpickle("./data/cifar-10-batches-py/batches.meta")
label_names = meta_data["label_names"]


X_train = np.concatenate((dict1["data"],dict2["data"],dict3["data"],dict4["data"],dict5["data"]))
y_train = np.concatenate((dict1["labels"],dict2["labels"],dict3["labels"],dict4["labels"],dict5["labels"]))
X_test = test["data"]
y_test = test["labels"]


# Define a range of regularization parameters (C is the inverse of regularization strength)
# We use a logarithmic scale from 10^-4 to 10^1
Cs = np.logspace(-4, 1, 10)

# Initialize LogisticRegressionCV with 4-fold cross-validation
# multi_class='multinomial' is used for CIFAR-10, and 'lbfgs' is a fast solver for large datasets
cv_model = LogisticRegressionCV(Cs=Cs, cv=4, solver='lbfgs', max_iter=500, n_jobs=-1)

print("Starting Cross-Validation...")
cv_model.fit(X_train, y_train)

# The scores_ attribute is a dict mapping each label to the grid of scores
# We average the scores across all folds and all classes to get the mean accuracy per C
mean_scores = np.mean([np.mean(scores, axis=0) for scores in cv_model.scores_.values()], axis=0)

# Make a log-log plot of the cross-validated accuracy score
plt.figure(figsize=(10, 6))
plt.loglog(Cs, mean_scores, marker='o')
plt.xlabel('Regularization Parameter C (log scale)')
plt.ylabel('Mean CV Accuracy (log scale)')
plt.title('4-Fold Cross-Validation Accuracy vs Regularization Parameter')
plt.grid(True, which="both", ls="-")
plt.show()

print(f"Best C found: {cv_model.C_[0]}")

# no convergence across all C values..
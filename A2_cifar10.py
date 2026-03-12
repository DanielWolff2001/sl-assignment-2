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

def data_to_image(x):
    return(x.reshape(3,32,32).transpose(1,2,0))

def plot_image(image, title=""):
    fig = plt.imshow(data_to_image(image))
    plt.title(title)
    fig.axes.set_axis_off()
    plt.show()

# as a verification that everything is working correctly, plot an image
# image_nr = 320
# plot_image(X_train[image_nr,:],label_names[y_train[image_nr]])

# large C means "Trust this training data a lot", overfitting
# small C says "This data may not be fully representative of the real world data", underfitting
# C_values = [0.001, 0.01, 0.1, 1, 10]
C_values = np.logspace(-3, 0, 8)


# Normalize the data to help with convergence.
X_test = X_test / 255
X_train = X_train / 255

train_times = []
test_accuracies = []

for C in C_values:

    start = time.time()

    model = LogisticRegression(C=C, max_iter=500, solver= 'lbfgs')
    model.fit(X_train, y_train)

    end = time.time()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)

    train_times.append(end - start)
    test_accuracies.append(acc)

    print("C =", C)
    print("Accuracy =", acc)
    print("Training time =", round(end - start, 2), "seconds")
    print()

plt.figure()

plt.semilogx(C_values, test_accuracies, marker='o')
plt.xlabel("Regularization parameter C")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs Regularization")
plt.grid(True)

plt.show()

plt.figure()

plt.semilogx(C_values, train_times, marker='o')
plt.xlabel("Regularization parameter C")
plt.ylabel("Training time (seconds)")
plt.title("Training Time vs Regularization")
plt.grid(True)

plt.show()

max_probs = np.max(y_prob, axis=1)


# Results solver sag and iter 200, C_values = [0.001, 0.01, 0.1, 1, 10]
# C = 0.001 did not converge within 200 iterations. Took 841.88 seconds and achieved an accuracy of 0.3841.
# C = 0.01 did not converge within 200 iterations. Took 863.7 seconds and achieved an accuracy of 0.3841.
# C = 0.1 did not converge within 200 iterations. Took 876.18 seconds and achieved an accuracy of 0.3839.
# C = 1 did not converge within 200 iterations. Took 884.16 seconds and achieved an accuracy of 0.384.
# C = 10 did not converge within 200 iterations. Took 942.96 seconds and achieved an accuracy of 0.3844.

# Results solver lbfgs and iter 500, C_values = [0.001, 0.01, 0.1, 1, 10]
# C = 0.001 did not converge within 500 iterations. Took 166.43 seconds and achieved an accuracy of 0.3896.
# C = 0.01 did not converge within 500 iterations. Took 168.16 seconds and achieved an accuracy of 0.3878.
# C = 0.1 did not converge within 500 iterations. Took 2035.78 seconds and achieved an accuracy of 0.3888.
# C = 1 did not converge within 500 iterations. Took 925.36 seconds and achieved an accuracy of 0.3927.
# C = 10 did not converge within 500 iterations. Took 170.28 seconds and achieved an accuracy of 0.3893.

# Results normalized data, solver lbfgs and iter 500, C_values = [0.001, 0.01, 0.1, 1]
# C = 0.001 converged within 500 iterations. Took 108.18 seconds and achieved an accuracy of 0.4047.
# C = 0.01 converged within 500 iterations. Took 138.55 seconds and achieved an accuracy of 0.4176.
# C = 0.1 did not converge within 500 iterations. Took 160.44 seconds and achieved an accuracy of 0.4068.
# C = 1 did not converge within 500 iterations. Took 175.94 seconds and achieved an accuracy of 0.3926.

# Results normalized data, solver lbfgs and iter 500, C_values = logspace(-3, 0, 8)
# See figures plotted. 
# WI4630 Statistical Learning – Assignment 2

This repository contains the code used for **Assignment 2** of the course *WI4630 Statistical Learning*.  
The assignment consists of two main parts: implementing logistic regression and applying regularized logistic regression to image datasets.

The report (PDF) contains the discussion of results and the code output. All code is also included in the appendix of the report.

---

## Repository Structure
.
├── A2_logistic_regression_simulation.py
├── A2_logistic_regression_mnist.py
├── A2_cifar10.py
├── mnist.csv
└── README.md

### Files

**A2_logistic_regression_simulation.py**

Code for **Question 1(a,b)**.

Structure:
- Implementation of logistic regression MLE using the **Newton–Raphson algorithm**
- Monte Carlo experiment for the simulated logistic regression model
- Computation of mean parameter estimates
- Histogram plots of the MLE for different sample sizes

---

**A2_logistic_regression_mnist.py**

Code for **Question 1(c,d)**.

Structure:
- Load MNIST data (digits 0 and 1)
- Run Newton–Raphson logistic regression
- Compute rank of the feature matrix
- Implement **ridge-regularized logistic regression**
- Evaluate predictive accuracy on the test set

---

**A2_cifar10.py**

Code for **Question 2(a–e)**.

Structure:
- Load CIFAR-10 dataset
- Split data into training and test sets
- Fit **regularized logistic regression (Scikit-Learn)**
- Cross-validation for the regularization parameter
- Evaluation using **accuracy** and **log-loss**

---

**mnist.csv**

Dataset used in Question 1 for binary classification of handwritten digits (0 vs 1).

---

## Requirements

Python ≥ 3.10

Packages:
- numpy
- pandas
- matplotlib
- scikit-learn

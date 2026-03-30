#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:21:05 2021

@author:  Jesse Vonk (5235960)
"""

#import packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

dir_path = os.path.dirname(os.path.realpath(__file__))


#read the csv file



df = pd.read_csv(fr'{dir_path}\\mnist.csv')
# Alternativelyyou can put the file in your working directory
# If you load the csv file with another function make sure that the matrix of features X is defined as in the book
# and the assignment and convert it to an numpy array



#make dataframe into numpy array
df.to_numpy()
y_labels_data=df['label'].to_numpy()
df_xdata=df.drop(columns='label')
x_features_data=df_xdata.to_numpy()
x_features_data.shape


# we will only use the zeros and ones in this empirical study
y_labels_01 = y_labels_data[np.where(y_labels_data <=1)[0]]
x_features_01 = x_features_data[np.where(y_labels_data <=1)[0]]



# create training set
n_train=100
y_train=y_labels_01 [0:n_train]
x_train=x_features_01[0:n_train]


#create test set
n_total=y_labels_01.size
y_test=y_labels_01 [n_train:n_total]
x_test=x_features_01[n_train:n_total]


 


 

## Here we plot some handwritten digits

plt.figure(figsize=(25,5))
for index, (image, label) in enumerate(zip(x_train[5:10], y_train[5:10])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.title('Label: %i\n' % label, fontsize = 20)
    
    
           

#The logistic function
def logistic(x):
    return 1 / (1 + np.exp(-x))


#given an estimated parameter for the logistic model we can obtain forecasts 
#for our test set with the following function

def logistic_forecast(features,beta):    
    signal_hat = np.dot(features, beta)
    y_hat=np.sign(signal_hat)
    y_hat[y_hat<0] = 0
    return y_hat
 

 
#computes the prediction error by comparing the predicted and the observed labels in the test set
def prediction_accuracy(y_predicted,y_observed):
    errors= np.abs(y_predicted-y_observed)
    total_errors=sum(errors)
    acc=1-total_errors/len(y_predicted)    
    return acc






#dimension of the problem
p=x_train.shape[1]

#Compute the ranks of the matrices X and X^T
print(np.linalg.matrix_rank(x_train))


def logistic_regression_NR(features, target, num_steps=100, tolerance=1e-6):
    beta = np.zeros(features.shape[1])

    for step in range(num_steps):     
        # Calculate probabilities p
        # p is the vector with elements p(xi; beta_old) 
        p = logistic(np.dot(features, beta))
        
        # compute gradient 
        # Score = X^T * (y - p) (from hastie et al)
        gradient = np.dot(features.T, target - p)
        
        # only update if gradient is large
        if np.linalg.norm(gradient) > tolerance:
            
            # compute Hessian
            # W is the diagonal matrix of weights p*(1-p) 
            # Hessian = -X^T * W * X (Hestie et al)
            W_diag = p * (1 - p)
            hessian = -np.dot(features.T, features * W_diag[:, np.newaxis])
            
            # Update beta according to Newton-Raphson procedure
            # beta_new = beta_old - (Hessian_inv * gradient) (from Hastie et al)
            beta = beta - np.dot(np.linalg.inv(hessian), gradient)
        else:
            # If the gradient is smaller than tolerance, we have converged
            break
          
    return beta

# Regularisation parameter
lambda_0=1


def logistic_regression_NR_penalized(features, target, lam, num_steps=100, tolerance=1e-6):
    p_dim = features.shape[1]
    beta = np.zeros(p_dim)
    
    for step in range(num_steps):     
        # Calculate probabilities p
        # p is the vector with elements p(xi; beta_old) 
        p = logistic(np.dot(features, beta))
        
        # compute gradient 
        # Score = X^T * (y - p) (from hastie et al)
        gradient = -np.dot(features.T, target - p) + 2 * lam * beta
        
        # only update if gradient is large
        if np.linalg.norm(gradient) > tolerance:
            
            # compute Hessian
            # W is the diagonal matrix of weights p*(1-p) 
            # Hessian = -X^T * W * X (Hestie et al)
            W_diag = p * (1 - p)
            hessian = np.dot(features.T, features * W_diag[:, np.newaxis]) + 2 * lam * np.eye(p_dim)
            
            # Update beta according to Newton-Raphson procedure
            # beta_new = beta_old - (Hessian_inv * gradient) (from Hastie et al)
            beta = beta - np.dot(np.linalg.inv(hessian), gradient)
        else:
            # If the gradient is smaller than tolerance, we have converged
            break
           
          
    return beta


if __name__ == '__main__':
    betas = logistic_regression_NR_penalized(x_train, y_train, lambda_0)
    
    y_pred = logistic_forecast(x_test, betas)
    test_acc = prediction_accuracy(y_pred, y_test)
    print(test_acc)
    


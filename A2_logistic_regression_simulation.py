
"""
Created on Mon Sep 20 12:11:28 2021

@author: Willem van den Bos 5323924
"""

#import packages

import numpy as np
import matplotlib.pyplot as plt

 
# n denotes the sample size 
n = 1000


# we simulate two types of features
half_sample=int(n/2)
x1 = np.random.multivariate_normal([-0.5, 1], [[1, 0.7],[0.7, 1]], half_sample)
x2 = np.random.multivariate_normal([2, -1], [[1, 0.7],[0.7, 1]], half_sample)
simulated_features = np.vstack((x1, x2)).astype(np.float64)


# the underlying value of beta in the simulation; the value we want to retrieve in the estimation procedure
beta_star=np.array([0.2,-0.8])



#The logistic function
def logistic(x):
    return 1 / (1 + np.exp(-x))


# Simulate the labels
def logistic_simulation(features,beta):    
    signal = np.dot(features, beta)
    p=logistic(signal)
    y= np.array([np.random.binomial(1, p[i] ) for i in range(n)])
    return y
 
simulated_labels = logistic_simulation(simulated_features, beta_star)

# #### Scatter plot of the features and correspoding labels
# plt.figure(figsize=(12,8))
# plt.scatter(simulated_features[:, 0], simulated_features[:, 1], c = simulated_labels, alpha = .5)
# plt.show()


#Skeleton for function Newton-Raphson for logisic regression

def logistic_regression_NR(features, target, num_steps=100, tolerance=1e-6):
    # initialization of beta
    beta = np.zeros(features.shape[1]) 
    
    for step in range(num_steps):     
        # Calculate probabilities p
        # p is the vector with elements p(xi; beta_old) 
        p = logistic(np.dot(features, beta))
        
        # compute gradient 
        # Score = X^T * (y - p) (from hastie et al)
        gradient = -np.dot(features.T, target - p)
        
        # only update if gradient is large
        if np.linalg.norm(gradient) > tolerance:
            
            # compute Hessian
            # W is the diagonal matrix of weights p*(1-p) 
            # Hessian = -X^T * W * X (Hestie et al)
            W_diag = p * (1 - p)
            hessian = np.dot(features.T, features * W_diag[:, np.newaxis])
            
            # Update beta according to Newton-Raphson procedure
            # beta_new = beta_old - (Hessian_inv * gradient) (from Hastie et al)
            beta = beta - np.dot(np.linalg.inv(hessian), gradient)
        else:
            # If the gradient is smaller than tolerance, we have converged
            break
                
    return beta


## Simulation study
S = 1000 
beta_estimates = []

for i in range(S):
    # Generate labels using the fixed features and true beta 
    y_sim = logistic_simulation(simulated_features, beta_star)
    
    # Estimate beta using Newton-Raphson [cite: 18, 20]
    beta_hat = logistic_regression_NR(simulated_features, y_sim)
    beta_estimates.append(beta_hat)

beta_estimates = np.array(beta_estimates)

# Calculate the average values to check against beta_star [cite: 13]
mean_beta1 = np.mean(beta_estimates[:, 0])
mean_beta2 = np.mean(beta_estimates[:, 1])

print("Mean of estimated beta_1:", mean_beta1)
print("Mean of estimated beta_2:", mean_beta2)

# Create one large figure for all three plots
plt.figure(figsize=(18, 5))

#Scatter plot of the simulated data
plt.subplot(1, 3, 1)
plt.scatter(simulated_features[:, 0], simulated_features[:, 1], c=simulated_labels, alpha=0.5)
plt.title('Simulated Feature Data')
plt.xlabel('x1')
plt.ylabel('x2')

#Histogram for Beta 1
plt.subplot(1, 3, 2)
plt.hist(beta_estimates[:, 0], bins=30, edgecolor='black', color='skyblue')
plt.axvline(beta_star[0], color='red', linestyle='--', label='True Beta 1')
plt.title('Distribution of Beta 1')
plt.legend()

#Histogram for Beta 2
plt.subplot(1, 3, 3)
plt.hist(beta_estimates[:, 1], bins=30, edgecolor='black', color='salmon')
plt.axvline(beta_star[1], color='red', linestyle='--', label='True Beta 2')
plt.title('Distribution of Beta 2')
plt.legend()

plt.tight_layout()
plt.show()




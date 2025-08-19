#!/usr/bin/env python3
"""
Code accompanying the paper: 
"Fast, Sample-Efficient, Affine-Invariant Private Mean and Covariance Estimation for Subgaussian Distributions"
Gavin Brown, Samuel Hopkins, and Adam Smith. COLT 2023.

This code is for research purposes only and should not be used in privacy-critical applications.

This script shows how to run the private mean and covariance estimation algorithms.
"""

import numpy as np
import time
from algorithms import (
    private_mean_estimation, 
    private_covariance_estimation, 
    stablecovariance
)
    
# Parameters
n, d = 100000, 10
eps, delta = 1.0, 1e-6 
lambda_0 = 50.0  # outlier threshold

# Generate data from N(0, I_d)
print("\nGenerating data from N(0, I_d)...")
np.random.seed(42)  # For reproducibility
X = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=n)

# Run private mean estimation
print("\nRunning private mean estimation...")
start_time = time.time()
private_mu = private_mean_estimation(X, eps, delta, lambda_0)
end_time = time.time()

if private_mu is not None:
    print(f"✓ Algorithm succeeded!")
    print(f"Private mean estimate: {private_mu}")
    print(f"L2 error: {np.linalg.norm(private_mu):.6f}")
else:
    print("✗ Algorithm failed (returned None)")
print(f"Runtime: {end_time - start_time:.3f} seconds")

# run private covariance estimation
print("\nRunning private covariance estimation...")
start_time = time.time()
private_Sigma = private_covariance_estimation(X, eps, delta, lambda_0)
end_time = time.time()

if private_Sigma is not None:
    print(f"✓ Algorithm succeeded!")
    # print(f"Private covariance estimate: {private_Sigma}")
    print(f"2-norm of (Sigma-Id): {np.linalg.norm(private_Sigma - np.eye(d),ord=2):.6f}")
else:
    print("✗ Algorithm failed (returned None)")
print(f"Runtime: {end_time - start_time:.3f} seconds")

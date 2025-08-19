"""
Code accompanying the paper: 
"Fast, Sample-Efficient, Affine-Invariant Private Mean and Covariance Estimation for Subgaussian Distributions"
Gavin Brown, Samuel Hopkins, and Adam Smith. COLT 2023.

This code is for research purposes only and should not be used in privacy-critical applications.

This script shows how to run the private mean and covariance estimation algorithms.
"""

import numpy as np

def split_and_center(X: np.ndarray) -> np.ndarray:
    """
    Splits X in half and returns (X[:m] - X[m:2*m]) / sqrt(2), where m = n // 2.
    """
    n = X.shape[0]
    m = n // 2
    return (X[:m] - X[m:2 * m]) / np.sqrt(2)


def PTR(z: float, eps: float, delta: float) -> str:
    """
    Implements the PTR mechanism from ptr_mechanism.tex.
    Args:
        z: the score (float or int)
        eps: privacy parameter epsilon
        delta: privacy parameter delta
    Returns:
        'PASS' or 'FAIL'
    """
    tau = 2 * np.log((1 - delta) / delta) / eps + 4
    if z == 0:
        return 'PASS'
    elif z >= tau:
        return 'FAIL'
    else:
        p_pass = 1 - np.exp(eps / 2 * (z - 2)) * delta
        if np.random.rand() < p_pass:
            return 'PASS'
        else:
            return 'FAIL' 

def leverage_scores(Y: np.ndarray) -> np.ndarray:
    """compute the leverage scores for all points in Y"""
    Q, R = np.linalg.qr(Y, mode='reduced')        # NumPy QR (no pivoting)
    diagR = np.abs(np.diag(R))
    
    r = int((diagR > tol).sum())                   # numerical rank
    Qr = Q[:, :r]
    return (Qr * Qr).sum(axis=1)

def pairwise_distances(X: np.ndarray, Y: np.ndarray, Sigma_inv: np.ndarray) -> np.ndarray:
    """compute the pairwise squared distances between all points in X and Y"""
    X_proj = X @ Sigma_inv            # shape: (n, d)
    Y_proj = Y @ Sigma_inv            # shape: (m, d)

    X_norm = np.einsum('ij,ij->i', X_proj, X)   # shape: (n,)
    Y_norm = np.einsum('ij,ij->i', Y_proj, Y)   # shape: (m,)

    # Step 2: Compute cross-term: -2 * x_i^T Sigma^{-1} y_j
    cross_term = X_proj @ Y.T                  # shape: (n, m)

    # Step 3: Combine
    D_squared = X_norm[:, None] - 2 * cross_term + Y_norm[None, :]  # shape: (n, m)
    
    # Ensure all squared distances are non-negative (handle numerical precision issues)
    D_squared = np.maximum(D_squared, 0)
    
    return D_squared
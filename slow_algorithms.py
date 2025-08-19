"""
Code accompanying the paper: 
"Fast, Sample-Efficient, Affine-Invariant Private Mean and Covariance Estimation for Subgaussian Distributions"
Gavin Brown, Samuel Hopkins, and Adam Smith. COLT 2023.

This code is for research purposes only and should not be used in privacy-critical applications.

This script shows how to run the private mean and covariance estimation algorithms.
"""

import numpy as np
from typing import Tuple, List, Set, Optional
from utils import PTR, split_and_center


def main_alg(
    X: np.ndarray,  # shape (n, d)
    eps: float,
    delta: float,
    lambda_0: float
) -> Optional[np.ndarray]:
    """
    Implements Algorithm: Private Mean Estimation, main_alg(X)
    """
    n, d = X.shape
    k = int(np.ceil(6*np.log(6/delta)/eps)+4)
    Rsize = 6 * k + int(np.ceil(18 * np.log(16 * n / delta)))
    c2 = 810 * lambda_0 * np.log(12 / delta) / (eps ** 2 * n ** 2)
    if n < 20 * k * lambda_0:
        return None  # FAIL

    # Sample R uniformly at random
    R = np.random.choice(n, size=Rsize, replace=False)

    # Check input size
    if n < 20 * k * lambda_0:
        return None  # FAIL

    # Compute nonprivate parameter estimates
    Y = split_and_center(X)
    hat_Sigma, score_1 = stablecovariance(Y, lambda_0, k)
    hat_mu, score_2 = stablemean(X, hat_Sigma, lambda_0, k, R)

    # Test the scores and release
    if PTR(max(score_1, score_2), eps / 3, delta / 6) == 'PASS':
        # Release noisy mean
        # this can be faster, generating a standard normal and then multiplying by the matrix
        noise = np.random.multivariate_normal(np.zeros(d), c2 * hat_Sigma)
        return hat_mu + noise
    else:
        return None  # FAIL

def slow_stablecovariance(
    Y: np.ndarray,  # shape (m, d)
    lambda_0: float,
    k: int
) -> Tuple[np.ndarray, int]:
    """
    Implements Algorithm: stablecovariance(Y, lambda_0, k)
    Y should be the output of split_and_center(X).
    """
    m, d = Y.shape
    S = []
    # Iterate over ells
    for ell in range(2 * k+1):
        lambda_ell = lambda_0 / (1 - (ell/(10*k)))
        S_ell = LargestGoodSubset(Y, lambda_ell)
        S.append(S_ell)
    
    score = min(k, min([m - len(S_ell) + ell for ell, S_ell in enumerate(S[:k + 1])]))
    w = np.zeros(m)
    for i in range(m):
        w[i] = sum(i in S_ell for S_ell in S[k + 1:]) / (k * m)
    hat_Sigma = np.zeros((d, d))
    for i in range(m):
        hat_Sigma += w[i] * np.outer(Y[i], Y[i])
    return hat_Sigma, score


def LargestGoodSubset(
    Y: np.ndarray,  # shape (m, d)
    lambda_: float
) -> Set[int]:
    """
    Implements Algorithm: LargestGoodSubset(Y, lambda)
    """
    m, d = Y.shape
    active_set = set(range(m))
    while True:
        # if not full rank, remove all points and return empty set
        if np.linalg.matrix_rank(Y[list(active_set)]) < d:
            return set()
        
        # compute the leverage scores for active points
        Y_active = Y[list(active_set)]
        Q, R = np.linalg.qr(Y_active, mode='reduced')
        leverage_scores = (Q[:, :d] * Q[:, :d]).sum(axis=1)

        # lambda thresholds correspond to rescaled leverage scores
        rescaled_norms = m * leverage_scores 
        
        # Find points to remove (leverage score > lambda_ell)
        active_list = list(active_set)
        points_to_remove = [active_list[i] for i, score in enumerate(rescaled_norms) if score > 2*lambda_]
        
        # If no points to remove, we're done
        if len(points_to_remove) == 0:
            return active_set
        
        # Else, remove outliers from active set
        active_set -= set(points_to_remove)


def slow_stablemean(
    X: np.ndarray,  # shape (n, d)
    hat_Sigma: np.ndarray,  # shape (d, d)
    lambda_0: float,
    k: int,
    R: np.ndarray  # indices, shape (Rsize,)
) -> Tuple[np.ndarray, int]:
    """
    Implements Algorithm: stablemean(X, hat_Sigma, lambda_0, k, R)
    """
    n, d = X.shape
    S = []
    gamma = (50/9) * (lambda_0 / n)
    # Iterate over ells
    for ell in range(2*k + 1):
        Lambda_ell = lambda_0 * ((1 - gamma) ** (-ell))
        tau = len(R) - ell
        S_ell = LargestCore(X, hat_Sigma, Lambda_ell, tau, R)
        S.append(S_ell)
        
    score = min(k, min([n - len(S_ell) + ell for ell, S_ell in enumerate(S[:k + 1])]))
    c = np.zeros(n)
    for i in range(n):
        c[i] = sum(i in S_ell for S_ell in S[k + 1:])
    Z = np.sum(c)
    w = c / Z if Z > 0 else np.zeros(n)
    hat_mu = np.sum(w[:, None] * X, axis=0)
    return hat_mu, score


def LargestCore(
    X: np.ndarray,  # shape (n, d)
    hat_Sigma: np.ndarray,  # shape (d, d)
    lambda_: float,
    tau: int,
    R: np.ndarray  # indices, shape (Rsize,)
) -> Set[int]:
    """
    Implements Algorithm: LargestCore(X, hat_Sigma, lambda, tau, R)
    """
    n, d = X.shape
    # Mahalanobis distance: (x_i - x_j)^T Sigma^{-1} (x_i - x_j)
    try:
        inv_Sigma = np.linalg.inv(hat_Sigma)
    except np.linalg.LinAlgError:
        # Singular matrix, treat all distances as infinity
        return set()
    
    # Extract R points
    X_R = X[R]  # shape (|R|, d)
    
    # Compute all pairwise differences: X[i] - X_R[j] for all i, j
    # This creates a (n, |R|, d) array
    diff = X[:, None, :] - X_R[None, :, :]  # shape (n, |R|, d)
    
    # Compute Mahalanobis distances efficiently
    # diff @ inv_Sigma @ diff.T would be (n, |R|, n, |R|), but we want (n, |R|)
    # So we compute: (diff @ inv_Sigma) * diff (element-wise)
    diff_inv_sigma = diff @ inv_Sigma  # shape (n, |R|, d)
    distances = np.sum(diff_inv_sigma * diff, axis=2)  # shape (n, |R|)
    
    # Count how many distances are <= lambda_ for each point i
    neighbor_counts = np.sum(distances <= lambda_, axis=1)  # shape (n,)
    
    # Return points that have at least tau neighbors
    result = set(np.where(neighbor_counts >= tau)[0])
    return result 
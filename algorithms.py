"""
Code accompanying the paper: 
"Fast, Sample-Efficient, Affine-Invariant Private Mean and Covariance Estimation for Subgaussian Distributions"
Gavin Brown, Samuel Hopkins, and Adam Smith. COLT 2023.

This code is for research purposes only and should not be used in privacy-critical applications.

This script shows how to run the private mean and covariance estimation algorithms.
"""


import numpy as np
from typing import Tuple, List, Set, Optional
from utils import PTR, split_and_center, pairwise_distances

def private_mean_estimation(
    X: np.ndarray,  # shape (n, d)
    eps: float,
    delta: float,
    lambda_0: float
) -> Optional[np.ndarray]:
    """
    Implements Private Mean Estimation, main algorithm of the paper.
    """
    n, d = X.shape
    k = int(np.ceil(6*np.log(6/delta)/eps)+4)
    Rsize = 6 * k + int(np.ceil(18 * np.log(16 * n / delta)))
    c2 = 810 * lambda_0 * np.log(12 / delta) / (eps ** 2 * n ** 2)
    
    print('min number of points required: ', 20 * k * lambda_0)
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

    print('score_1: ', score_1)
    print('score_2: ', score_2)
    # Test the scores and release
    if PTR(max(score_1, score_2), eps / 3, delta / 6) == 'PASS':
        # Release noisy mean
        # this can be faster, generating a standard normal and then multiplying by the matrix
        noise = np.random.multivariate_normal(np.zeros(d), c2 * hat_Sigma)
        return hat_mu + noise
    else:
        return None  # FAIL

def stablecovariance(
    Y: np.ndarray,  # shape (m, d)
    lambda_0: float,
    k: int
) -> Tuple[np.ndarray, int]:
    """
    Fast implementation of stablecovariance using collapsing approach.
    Instead of calling LargestGoodSubset multiple times, we maintain
    a single active set and progressively remove outliers.

    As described in the paper, this can be done asymptotically faster using the fact that
    all outlier removals cause only rank-one updates, which mean we can compute updated quantities
    faster via the Sherman-Morrison formula.
    """
    m, d = Y.shape
    S = []
    
    # Start with all points active
    active_set = set(range(m))
    
    # Iterate in decreasing order of ell (largest threshold first)
    for ell in range(2 * k, -1, -1):
        lambda_ell = lambda_0 / (1 - (ell/(10*k)))
        
        # Iteratively remove outliers at this threshold
        while True:
            # check that the covariance is full rank; if not, remove all active points
            cov = Y[list(active_set)].T @ Y[list(active_set)]
            if np.linalg.matrix_rank(cov) < d:
                active_set = set()
                break
            
            # compute the leverage scores for active points
            Y_active = Y[list(active_set)]
            Q, R = np.linalg.qr(Y_active, mode='reduced')
            leverage_scores = (Q[:, :d] * Q[:, :d]).sum(axis=1)

            # outlier threshold lives on a different scale than the leverage scores
            rescaled_norms = m * leverage_scores 
            
            # Find points to remove (norms > lambda_ell)
            active_list = list(active_set)
            points_to_remove = [active_list[i] for i, score in enumerate(rescaled_norms) if score > lambda_ell]
            
            # If no points to remove, we're done at this threshold
            if len(points_to_remove) == 0:
                break
            
            # Remove outliers from active set
            active_set -= set(points_to_remove)
        
        # append current active set to S
        S.append(active_set)
        
        # Early stopping: if |active_set| <= m - k, we can stop
        if len(active_set) <= m - k:
            # Fill remaining slots with empty sets
            # the slots are 0, 1, 2, ..., ell-1; in reverse order
            for _ in range(ell):
                S.append(set())
            break
    
    # Reverse S to maintain original order (ell=0, 1, 2, ..., 2k)
    S = S[::-1]
    
    # Compute score and weights
    score = min(k, min([m - len(S_ell) + ell for ell, S_ell in enumerate(S[:k + 1])]))
    w = np.zeros(m)
    for i in range(m):
        w[i] = sum(i in S_ell for S_ell in S[k + 1:]) / (k * m)
    
    # Compute weighted covariance (vectorized)
    # hat_Sigma = sum_i w[i] * Y[i] @ Y[i].T
    # This is equivalent to Y.T @ diag(w) @ Y
    hat_Sigma = Y.T @ (w[:, None] * Y)
    
    return hat_Sigma, score

def stablemean(
    X: np.ndarray,  # shape (n, d)
    hat_Sigma: np.ndarray,  # shape (d, d)
    lambda_0: float,
    k: int,
    R: np.ndarray  # indices, shape (Rsize,)
) -> Tuple[np.ndarray, int]:
    """
    Fast implementation of stablemean using pre-computed boolean matrix.
    Compute the rescaled Mahalanobis distances, then compute the set inclusion matrix.
    After that, scores and weights are easy to compute.
    """
    n, d = X.shape
    gamma = (50/9) * (lambda_0 / n)
    
    # Compute inverse of hat_Sigma once
    try:
        inv_Sigma = np.linalg.inv(hat_Sigma)
    except np.linalg.LinAlgError:
        # Singular matrix, return empty result
        return np.zeros(d), k
    
    # Extract R points
    X_R = X[R]  # shape (|R|, d)
    
    # Compute all pairwise Mahalanobis distances once
    # This creates a (n, |R|) matrix of distances
    distances = pairwise_distances(X, X_R, inv_Sigma)
    
    # tracks which points are in which good subsets
    # set_inclusion[i] = smallest ell such that i\in S_ell, -1 if i\in no good subset
    set_inclusion = compute_set_inclusion(distances, lambda_0, k)
    
    # Compute score
    set_sizes = np.bincount(set_inclusion, minlength=2*k+1)
    set_sizes = set_sizes[:k+1] # only use the first k+1 sets to compute score
    scores = n - set_sizes + np.arange(k+1)  # shape (k+1,)
    score = min(k, np.min(scores))
    
    # Compute weights
    # proportional to the number of good subsets (k+1 or above) that it appears in
    c = np.minimum(set_inclusion - k, 0)
    Z = np.sum(c)
    w = c / Z if Z > 0 else np.zeros(n)
    
    # Compute weighted mean
    hat_mu = np.sum(w[:, None] * X, axis=0)
    
    return hat_mu, score 

def compute_set_inclusion(
    distances: np.ndarray,  # shape (n, |R|)
    lambda_0: float,
    k: int
) -> np.ndarray:
    """
    Subroutine for fast stablemean.

    Given the pairwise distances between each the points and references points, 
    for each point i in [n], find the smallest ell such that i in S_ell.

    Args:
        distances: np.ndarray, shape (n, |R|), the distances between each point and the reference points
        lambda_0: float, the initial threshold
        k: int; we have 2*k+1 good subsets total
    """
    n, len_R = distances.shape
    gamma = (50/9) * (lambda_0 / len_R)
    
    # set zero distances to lambda_0/2, so we can take logs and avoid division by zero
    # but all the distances are smaller than lambda_0, so this doesn't change the result
    distances[distances == 0] = lambda_0 / 2

    # what is the smallest ell such that the distance is less than lambda_ell := lambda_0 / (1 - gamma)^ell?
    log_ratio = np.log(lambda_0) - np.log(distances/2)
    log_factor = np.log(1 - gamma)
    computed_indices = np.ceil(log_ratio / log_factor).astype(int)
    
    # all indices should be at least 0
    # indices larger than 2k mean that the distance is too large, the points are not neighbors for any threshold.
    computed_indices = np.maximum(computed_indices, 0)

    # For each i, what is the smallest ell such that i\in S_ell?
    set_inclusion = np.full(n, -1, dtype=int)  # -1 means not in any good subset

    for i in range(n):
        # Get threshold indices for this point
        point_thresholds = computed_indices[i, :]
        
        # Count neighbors at each threshold level using bincount
        neighbor_counts = np.bincount(point_thresholds, minlength=2 * k + 1)
        neighbor_counts = neighbor_counts[:2*k + 1] # don't care about larger distances

        # Compute cumulative neighbor counts
        cumulative_counts = np.cumsum(neighbor_counts)
        
        # Find transition point: first threshold where cumulative_count >= tau
        tau_values = len_R - np.arange(2 * k + 1)
        transition_mask = cumulative_counts >= tau_values
        if np.any(transition_mask):
            set_inclusion[i] = np.argmax(transition_mask) # ie, first occurrence of True
    
    return set_inclusion

def private_covariance_estimation(
    X: np.ndarray,  # shape (n, d)
    eps: float,
    delta: float,
    lambda_0: float
) -> Optional[np.ndarray]:
    """
    Implements Algorithm: Private Covariance Estimation, A_cov^eps,delta,lambda_0(x)
    """
    n, d = X.shape
    k = int(np.ceil(4 * np.log(2 / delta) / eps) + 4)
    N = int(np.floor((1 / 24) * (n ** 2 * eps ** 2) / (lambda_0 ** 2 * np.log(2 / delta))))
    
    # Check input size requirement
    if n < 10 * k * lambda_0:
        return None  # FAIL
    
    # Compute nonprivate covariance estimate
    hat_Sigma, score = stablecovariance(X, lambda_0, k)
    
    # Test the score and release
    if PTR(score, eps / 2, delta / 2) == 'PASS':
        # Draw N iid samples from N(0, hat_Sigma)
        Z_samples = np.random.multivariate_normal(np.zeros(d), hat_Sigma, size=N)
        
        # Compute the empirical covariance of the samples
        # This is equivalent to (1/N) * sum_{i=1}^N Z_i Z_i^T
        result = np.zeros((d, d))
        for i in range(N):
            result += np.outer(Z_samples[i], Z_samples[i])
        result /= N
        
        return result
    else:
        return None  # FAIL 
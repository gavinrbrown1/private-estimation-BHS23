# BHS23 Algorithm Implementation

Code accompanying the paper: 
**"Fast, Sample-Efficient, Affine-Invariant Private Mean and Covariance Estimation for Subgaussian Distributions"**
by Gavin Brown, Samuel Hopkins, and Adam Smith. COLT 2023.
[paper on arXiv](https://arxiv.org/abs/2301.12250)

**Research Code Warning**: This code is for research purposes only and should not be used in privacy-critical applications.

## Overview

This repository implements algorithms for differentially private mean estimation and covariance estimation from the BHS23.
The code uses NumPy (tested with version 2.3.2).
The script `demo.py` generates synthetic data and runs the mean and covariance estimators.
The main code lives in `algorithms.py`, with additional functions in `utils.py`.

The paper's pseudocode presentation is optimized for readability and checkability, not computational efficiency.
The code in `algorithms.py` is an equivalent but faster implementation.
To help bridge this gap, we provide another file, `slow_algorithms.py`, containing implementations that follow the pseudocode nearly line-by-line.
For further discussion, see Section 2.3 "Running Time" in the paper.

## Usage Details

Run the demonstration script to see the algorithms in action:

```bash
python demo.py
```

This will:
- Generate synthetic data from a multivariate normal distribution
- Run private mean estimation
- Run private covariance estimation
- Display results and timing information

The call to the private mean estimation subroutine looks like this:
```python
private_mu = private_mean_estimation(X, eps, delta, lambda_0)
```
Here `X` is the dataset, `eps` and `delta` are privacy parameters, and `lambda_0` is a user-supplied outlier threshold.
The algorithm is `(eps,delta)`-DP for any setting of `lambda_0`, but we need to set it appropriately to ensure accuracy. 
Here are the main tradeoffs:
- If `lambda_0` is very large, the algorithm adds a lot of noise.
- If `lambda_0` is too small, the algorithm removes too many points as outliers and returns `FAIL`.
- The algorithm checks if `n` is sufficiently large relative to `lambda_0`, and only proceeds if this check passes.
Setting `lambda_0` appropriately will require some trial-and-error. 
This itself could be privatized, e.g., by finding the smallest `lambda_0` where the algorithm does not fail. 
Asymptotically, setting `lambda_0` to be roughly `d + O(log n)` suffices for Gaussian data; see the paper for further theoretical discussion.

The algorithms require a certain number of data points in order to ensure privacy. 
(Both have a line of the form "`if n < (...) then FAIL`.)
The exact threshold depends on `lambda_0` and the privacy parameters.
It is a substantial practical drawback: in `demo.py` we run with `d=10` dimensions, `eps=1`, `delta=1e-6`, and `lambda_0=50`; 
in this setting, the private mean estimation algorithm will automatically fail if it receives an input with fewer than `n=98000` examples.

The discussion for private covariance estimation is similar.



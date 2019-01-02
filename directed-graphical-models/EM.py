"""My implementation of the EM algorithm for Gaussian Mixture Model based on the
implementation in the scikit-learn package, with all the Cholesky factorization
taken out because I've made assumptions about the covariance matrix, so 
I can directly sample from a multivariate normal distribution specified using
the scipy.stats package. 

I like elegance code, which is missing in other examples. Thi way I can directly 
write down the code by looking at the algorithm. 

It seems to give quite decent results. However I haven't performed wrapped
it up in a class or performed decent tests."""

import numpy as np
from scipy.stats import multivariate_normal

def _estimate_gaussian_covariances(resp, X, nk, means):
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[::n_features + 1] += 1.e-6
    return covariances


def _estimate_gaussian_parameters(X, resp):
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = _estimate_gaussian_covariances(resp, X, nk, means)
    
    return nk, means, covariances


def _estimate_log_gaussian_prob(X, means, cov):
    """This uses the scipy.stat.smultivariate_normal class, might cause problems 
    if the covariance matrix is not positive semi-definite. """
    rv = multivariate_normal(means, cov)
    return np.log(rv.pdf(X))


def _estimate_log_prob(X, means, covariances):
    """Estimate log probability.
    Parameters
    ----------
    X : array-like, shape (n_samples, 2)
    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, _ = X.shape
    n_components, _ = means.shape
    
    log_prob = np.empty([n_samples, n_components])
    
    for k in range(n_components):
        log_prob[:, k] = _estimate_log_gaussian_prob(X, means[k], covariances[k])    
    return log_prob


def _estimate_log_weights(weights_):
    return np.log(weights_)


def _estimate_log_prob_resp(X, weights, means, covariances):
    """Estimate log probabilities and responsibilities for each sample.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, 2)
    
    weights: array-like, shape (n_components,)
    
    means: array-like, shape (n_components, 2)
    
    covariances: array-like, shape (n_components, 2, 2)
    
    Returns
    -------
    log_prob_norm : array, shape (n_samples,)
    
    log_responsibilities : array, shape (n_samples, n_components)
        logarithm of the responsibilities
    """
    log_prob = _estimate_log_prob(X, means, covariances)
    
    weighted_log_prob = log_prob + _estimate_log_weights(weights)   
    
    log_prob_norm = np.log(np.sum(np.exp(weighted_log_prob), axis=1))
        
    log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
    
    return log_prob_norm, log_resp
                          

def _estimate_log_likelihood(X, weights, means, covariances):
    """Estimate log likelihood """
    n_samples, _ = X.shape
    n_components, _ = means.shape
    
    log_prob = _estimate_log_prob(X, means, covariances)
    return np.sum(np.log(np.exp(log_prob).dot(weights)))
                          

def _e_step(X, weights, means, covariances):
    """E step.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    Returns
    -------
    log_prob_norm : float
        Mean of the logarithms of the probabilities of each sample in X
    log_responsibility : array, shape (n_samples, n_components)
        Logarithm of the posterior probabilities (or responsibilities) of
        the point of each sample in X.
    """
    log_prob_norm, log_resp = _estimate_log_prob_resp(X, weights, means, covariances)
    
    return np.mean(log_prob_norm), log_resp
                          

def _m_step(X, log_resp):
    """M step.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    log_resp : array-like, shape (n_samples, n_components)
        Logarithm of the posterior probabilities (or responsibilities) of
        the point of each sample in X.
    """
    n_samples, _ = X.shape
    weights_, means_, covariances_ = (_estimate_gaussian_parameters(X, np.exp(log_resp)))
    
    weights_ /= n_samples
    return weights_, means_, covariances_                          
   
def _compute_lower_bound(_, log_prob_norm):
    return log_prob_norm

# Functions to generate the test example
def gen_pos_def_matrix(size):
    """Generate a random positive-definite matrix for MVN sampling. 
    This can be simply achieved by multiplying a matrix with its transpose.
    See wikipedia article.
    http://en.wikipedia.org/wiki/Positive-definite_matrix#Negative-definite.2C_semidefinite_and_indefinite_matrices
    """
    A = np.random.randint(-10, 10, size=(size, size))*0.1 # Let the values not be too large
    return A.dot(A.T)

def create_sample(n):
    """Here we sample from three 2x2 multivariate Gaussians, 
    multiply them by the given weights and 
    combine them to return a 3n x 2 matrix for inference.
    """
    mu1, sigma1 = [0.2, 0.5], [[0.53, 0.51], [0.51, 0.89]]
    mu2, sigma2 = [0.5, 0.5], [[0.34, 0.63], [0.63, 1.17]]
    mu3, sigma3 = [0.75, 0.5], [[ 0.13, -0.29], [-0.29,  1.09]]
    
    weights = [0.2, 0.3, 0.5] # weights must add up to 1
    
    # Sampling
    X1 = np.random.multivariate_normal(mu1, sigma1, 2*n)
    X2 = np.random.multivariate_normal(mu2, sigma2, 3*n)
    X3 = np.random.multivariate_normal(mu3, sigma3, 5*n)
    
    act_weights = np.array(weights)
    act_means =  np.array([mu1, mu2, mu3])
    act_covs = np.array([sigma1, sigma2, sigma3])
        
    return np.concatenate((X1, X2, X3), axis=0), act_weights, act_means, act_covs                          


if __name__ == "__main__":
    # Initial guesses
    weights = np.array([0.3, 0.3, 0.4])
    means = np.array([[0.2, 0.2], [0.3, 0.5], [0.2, 0.8]])
    variances = np.array([np.eye(2)*0.5, np.eye(2)*0.5, np.eye(2)*0.5])

    print('actual:', act_weights, act_means, act_covs)
    
    # Sample from the actual distributions
    X, act_weights, act_means, act_covs = create_sample(100)
    
    # Run the EM algorithm
    max_iter = 100
    lower_bound = -np.infty
    tol = 1.e-4

    log_likelihood = []

    for n_iter in range(1, max_iter + 1):
        prev_lower_bound = lower_bound
        log_prob_norm, log_resp = _e_step(X, weights, means, variances)
        weights, means, variances = _m_step(X, log_resp)
        
        lower_bound = _compute_lower_bound(log_resp, log_prob_norm)
        change = lower_bound - prev_lower_bound

        if abs(change) < tol:
            print('Converged')
            print('Weights:', weights, 'Means:', means, 'covariances:', variances)
            break
            
# With this set of initial values we get this, which is remarkably good actually.
# Weights: [0.21002659 0.29206699 0.49790642] 

# Means: [[0.27103788 0.33794394]
# [0.4705445  0.44514773]
# [0.79662441 0.72473925]] 

#covariances: [[[ 0.50830226  0.53661749]
#  [ 0.53661749  0.93529743]]

# [[ 0.31714016  0.58512662]
#  [ 0.58512662  1.08223355]]

# [[ 0.13431471 -0.3015155 ]
#  [-0.3015155   1.15939371]]]            
    
                          
                          
                          
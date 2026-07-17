import numpy as np

def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    
    Parameters:
    - prob_distributions: 2D array-like of shape (sequence_length, vocab_size)
    - actual_tokens: 1D array-like of token IDs of shape (sequence_length,)
    """
    # Convert inputs to numpy arrays for efficient indexing
    probs = np.asarray(prob_distributions)
    tokens = np.asarray(actual_tokens)
    
    # 1. Extract the predicted probability for each actual token
    # Uses advanced indexing: row indices [0, 1, 2...] and column indices from tokens
    target_probabilities = probs[np.arange(len(tokens)), tokens]
    
    # 2. Compute the negative log-likelihood for each token
    # A small epsilon prevents log(0) errors if a probability is exactly zero
    eps = 1e-15
    neg_log_likelihoods = -np.log(np.clip(target_probabilities, eps, 1.0))
    
    # 3. Calculate the mean log-likelihood
    mean_nll = np.mean(neg_log_likelihoods)
    
    # 4. Perplexity is the exponential of the mean cross-entropy loss
    perp = np.exp(mean_nll)
    
    return float(perp)

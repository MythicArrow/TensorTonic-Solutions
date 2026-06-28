import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    
    Args:
        Z1: Embeddings of shape (batch_size, embedding_dim)
        Z2: Embeddings of shape (batch_size, embedding_dim)
        temperature: Scaling factor for the logits
    
    Returns:
        Average InfoNCE loss over the batch
    """
    Z1 = np.stack(Z1)
    Z2 = np.stack(Z2)
    batch_size = Z1.shape
    
    # 1. Compute similarity matrix (cosine similarity or dot product)
    # Shape: (batch_size, batch_size)
    # Z1[i] compared against all Z2[j]
    logits = np.dot(Z1, Z2.T) / temperature
    
    # 2. Numerical stability: subtract max per row
    # This prevents exp() overflow
    logits = logits - np.max(logits, axis=1, keepdims=True)
    
    # 3. Extract positive pairs (diagonal of the matrix)
    # Shape: (batch_size,)
    #labels = np.arange(batch_size)
    positives = np.diag(logits)
    
    # 4. Compute the denominator: sum of exp(logits) for ALL samples in the row
    # Shape: (batch_size,)
    log_sum_exp = np.log(np.sum(np.exp(logits), axis=1))
    
    # 5. Calculate loss per sample: - (positive - log_sum_exp)
    # This is equivalent to -log(exp(pos) / sum(exp(logits)))
    loss_per_sample = positives - log_sum_exp
    
    # 6. Average loss over the batch
    return -np.mean(loss_per_sample)
    
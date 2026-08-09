import numpy as np

def silhouette_score(X, labels):
    """
    Computes the mean Silhouette Coefficient of all samples using NumPy.
    
    Parameters:
    -----------
    X : ndarray of shape (n_samples, n_features)
        Feature array.
    labels : ndarray of shape (n_samples,)
        Cluster labels for each sample.
        
    Returns:
    --------
    float
        The mean silhouette score across all samples.
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2 or n_clusters >= n_samples:
        raise ValueError("The number of clusters must be between 2 and n_samples - 1.")

    # 1. Compute the full pairwise Euclidean distance matrix using broadcasting
    # (xi - xj)^2 = xi^2 - 2*xi*xj + xj^2
    X_square = np.sum(X**2, axis=1, keepdims=True)
    dist_matrix = np.sqrt(np.maximum(X_square - 2 * np.dot(X, X.T) + X_square.T, 0))

    # Initialize arrays to store a(i) and b(i)
    a = np.zeros(n_samples)
    b = np.full(n_samples, np.inf)

    # 2. Vectorize the calculation of a(i) and b(i)
    for label in unique_labels:
        # Create boolean mask for the current cluster
        in_cluster = (labels == label)
        cluster_size = np.sum(in_cluster)
        
        if cluster_size <= 1:
            a[in_cluster] = 0.0
            continue
            
        # Extract distances from samples in this cluster to all other samples
        cluster_dists = dist_matrix[in_cluster]

        # a(i): Mean distance to other points in the same cluster
        # Sum rows of the submatrix matching the cluster and divide by (size - 1)
        a[in_cluster] = np.sum(cluster_dists[:, in_cluster], axis=1) / (cluster_size - 1)

        # b(i): Mean distance to the closest neighboring cluster
        for other_label in unique_labels:
            if other_label == label:
                continue
            other_cluster = (labels == other_label)
            other_size = np.sum(other_cluster)
            
            if other_size == 0:
                continue
                
            # Mean distance from current cluster points to all points in 'other_cluster'
            mean_inter_dist = np.mean(cluster_dists[:, other_cluster], axis=1)
            
            # Keep the minimum average distance to another cluster
            b[in_cluster] = np.minimum(b[in_cluster], mean_inter_dist)

    # 3. Calculate silhouette coefficients for each sample
    denominator = np.maximum(a, b)
    
    # Handle the edge case where denominator is 0 (isolated points)
    with np.errstate(divide='ignore', invalid='ignore'):
        silhouette_samples = np.where(denominator > 0, (b - a) / denominator, 0)

    # 4. Return the overall mean silhouette score
    return np.mean(silhouette_samples)

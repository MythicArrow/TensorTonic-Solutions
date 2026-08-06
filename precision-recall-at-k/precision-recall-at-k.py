def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    top_k = recommended[:k]
    relevant_set = set(relevant)

    # Count how many recommended items in the top k are relevant
    count = sum(1 for item in top_k if item in relevant_set)

    precision = count / k if k > 0 else 0
    recall = count / len(relevant_set) if relevant_set else 0

    return_list = [precision, recall]

    return return_list

    
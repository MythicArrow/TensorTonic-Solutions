def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    # Your code here
    s = set(stopwords)
    filtered = [word for word in tokens if word not in s]
    return filtered
    
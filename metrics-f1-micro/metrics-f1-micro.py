def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Return the micro-averaged F1 score rounded to four decimals.
    """
    # Write code here
    tp = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))
    fp = sum(actual != predicted for actual, predicted in zip(y_true, y_pred))
    micro = (2 * tp) / (2 * tp + fp + fp)
    return micro
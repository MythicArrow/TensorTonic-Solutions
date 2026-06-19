import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    a = np.atleast_2d(np.asarray(a, dtype=float))   # (D,) -> (1, D), (N,D) kalır
    b = np.atleast_2d(np.asarray(b, dtype=float))
    y = np.asarray(y, dtype=float).ravel()          # (N,)

    d = np.linalg.norm(a - b, axis=1)               # (N,) öklid mesafe
    loss = y * d**2 + (1 - y) * np.maximum(0.0, margin - d)**2

    if reduction == "sum":
        return float(np.sum(loss))
    return float(np.mean(loss))
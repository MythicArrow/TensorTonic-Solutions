import numpy as np

def clip_gradients(g, max_norm):
    """Clip gradients using global norm clipping."""
    g = np.asarray(g)
    
    
    l2_norm = np.linalg.norm(g)
    
    
    if l2_norm > max_norm and l2_norm > 0 and max_norm > 0:
        return g * (max_norm / l2_norm)
        
    return g.copy()

import math

def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    """
    Compute the learning rate using cosine annealing.
    
    Formula: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * current_step / total_steps))
    """
    # Calculate the cosine term
    # Ensure we don't divide by zero if total_steps is 0
    if total_steps == 0:
        return float(min_lr)
        
    cos_val = math.cos(math.pi * current_step / total_steps)
    
    # Standard Cosine Annealing Formula
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos_val)
    
    return float(lr)

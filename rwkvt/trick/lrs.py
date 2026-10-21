import math


def cos_decay(initial_lr, final_lr, current_step, total_steps):
    """
    Compute the cosine decayed learning rate with a final learning rate.

    Args:
        initial_lr (float): Initial learning rate.
        final_lr (float): Final learning rate.
        current_step (int): Current training step.
        total_steps (int): Total number of training steps in one epoch.

    Returns:
        float: Decayed learning rate.
    """
    if current_step>=total_steps:
        lr = final_lr
    else:
        lr = final_lr + 0.5 * (initial_lr - final_lr) * (1 + math.cos(math.pi * current_step / total_steps))
    return lr

def wsd(initial_lr, final_lr, current_step, total_steps, warmup_steps=100):
    """
    Compute the learning rate using cosine annealing with a warmup phase.

    Warmup phase:
        For the first warmup_steps, the learning rate increases linearly from 0 to initial_lr.
    Cosine annealing phase:
        From warmup_steps to total_steps, the learning rate decays from initial_lr to final_lr
        following the cosine annealing schedule.

    Args:
        initial_lr (float): The target learning rate after warmup (also the starting learning rate for decay).
        final_lr (float): The final learning rate after total_steps.
        current_step (int): Current training step.
        total_steps (int): Total number of training steps.
        warmup_steps (int): Number of steps used for the warmup phase.

    Returns:
        float: The computed learning rate.
    """
    if warmup_steps<=0:
        warmup_steps = 100
    if current_step < warmup_steps:
        # Warmup phase: linearly increase LR from 0 to initial_lr.
        return initial_lr * current_step / max(1, warmup_steps)
    else:
        # Adjust step count for cosine annealing phase.
        effective_step = current_step - warmup_steps
        effective_total = total_steps - warmup_steps

        if effective_step >= effective_total:
            return final_lr

        # Compute cosine annealing decay.
        cosine_decay = 0.5 * (1 + math.cos(math.pi * effective_step / effective_total))
        decayed_lr = final_lr + (initial_lr - final_lr) * cosine_decay
        return decayed_lr

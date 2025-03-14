import math

def compute_zcdp(q, noise_multiplier, steps):
    """
    Computes the zCDP (rho) for the Sampled Gaussian Mechanism.

    Args:
        q (float): Sampling probability, generally batch_size / number_of_samples.
        noise_multiplier (float): Noise multiplier (sigma) for DP-SGD.
        steps (int): Number of iterations the mechanism is applied.

    Returns:
        float: The zCDP rho value.
    """
    sigma = noise_multiplier
    rho = steps * (q ** 2) / (2 * sigma ** 2)
    return rho

def get_privacy_spent(rho, target_delta):
    """
    Converts zCDP rho to epsilon-delta DP parameters.

    Args:
        rho (float): zCDP rho parameter.
        target_delta (float): The target delta.

    Returns:
        tuple: epsilon, delta, and optimal order (rho).
    """
    if target_delta is None:
        raise ValueError("target_delta must be provided for zCDP conversion.")
    
    epsilon = rho + 2 * math.sqrt(rho * math.log(1 / target_delta))
    return epsilon, target_delta, rho

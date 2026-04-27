"""Sobol quasi-random normal generation for MC path simulation.

Generates low-discrepancy normal samples using scrambled Sobol sequences
with inverse CDF transform. 

Includes a full Brownian Bridge construction to conquer the curse of 
dimensionality in high-step/multi-asset American Monte Carlo, guaranteeing 
variance reduction even at 256+ dimensions.
"""
import math
import torch


def generate_sobol_normals(
    num_assets: int,
    num_steps: int,
    num_paths: int,
    scramble: bool = True,
    device: str = "cuda",
    skip: int = 0,
    use_brownian_bridge: bool = True,
) -> torch.Tensor:
    """Generate quasi-random standard normals via Sobol + inverse CDF.

    Args:
        num_assets:          Number of correlated assets (N).
        num_steps:           Number of time steps (T).
        num_paths:           Number of MC paths (P).
        scramble:            Whether to apply Owen scrambling.
        device:              Target device.
        skip:                Number of paths to fast-forward past (for out-of-sample).
        use_brownian_bridge: Restructures dimension mapping to preserve QMC advantage 
                             in high-dimensional pricing.

    Returns:
        Z: float32 tensor [N, T, P] of standard normal increments.
    """
    dim = num_assets * num_steps
    engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=scramble)

    if skip > 0:
        engine.fast_forward(skip)

    uniforms = engine.draw(num_paths)
    uniforms = uniforms.clamp(1e-6, 1.0 - 1e-6)

    normals = torch.erfinv(2.0 * uniforms - 1.0) * math.sqrt(2.0)
    normals = normals.to(device=device, dtype=torch.float32)
    normals = normals.reshape(num_paths, num_steps, num_assets).permute(2, 1, 0).contiguous()

    if use_brownian_bridge:
        normals = _apply_brownian_bridge(normals)

    return normals


def generate_sobol_normals_1d(
    num_steps: int,
    num_paths: int,
    scramble: bool = True,
    device: str = "cuda",
    skip: int = 0,
    use_brownian_bridge: bool = True,
) -> torch.Tensor:
    """Generate quasi-random normals for single-asset (vanilla) options.

    Returns:
        Z: float32 tensor [T, P] of standard normal increments.
    """
    return generate_sobol_normals(
        1, num_steps, num_paths, scramble, device, skip, use_brownian_bridge
    ).squeeze(0)


def _apply_brownian_bridge(normals: torch.Tensor) -> torch.Tensor:
    """Transforms sequential standard normals into Brownian Bridge increments.
    
    By mapping the lowest, highest-quality Sobol dimensions to the widest 
    time intervals (T, then T/2, then T/4), we heavily reduce the variance 
    of the terminal payoff.
    """
    N, T, P = normals.shape
    
    W = torch.zeros((N, T + 1, P), dtype=normals.dtype, device=normals.device)
    
    queue = [(0, T)]
    bridge_ops = []
    while queue:
        left, right = queue.pop(0)
        if right - left > 1:
            mid = (left + right) // 2
            bridge_ops.append((left, mid, right))
            queue.append((left, mid))
            queue.append((mid, right))
            
    W[:, T, :] = normals[:, 0, :] * math.sqrt(T)
    
    for i, (left, mid, right) in enumerate(bridge_ops):
        dim_idx = i + 1
        
        weight_left = (right - mid) / (right - left)
        weight_right = (mid - left) / (right - left)
        std_dev = math.sqrt((mid - left) * (right - mid) / (right - left))
        
        W[:, mid, :] = (
            weight_left * W[:, left, :] + 
            weight_right * W[:, right, :] + 
            normals[:, dim_idx, :] * std_dev
        )
        
    increments = W[:, 1:, :] - W[:, :-1, :]
    
    return increments.contiguous()

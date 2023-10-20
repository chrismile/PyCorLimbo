from __future__ import annotations
import torch
import pycorlimbo
import typing

__all__ = [
    "optimize_single_threaded",
    "optimize_multi_threaded"
]


class BayOptSettings:
    def __init__(self) -> None: ...
    xs: int = ...
    ys: int = ...
    zs: int = ...
    num_init_samples: int = ...
    num_iterations: int = ...
    num_optimizer_iterations: int = ...
    alpha: float = ...

def optimize_single_threaded(settings: BayOptSettings, sample_tensor: torch.Tensor, callback: typing.Callable) -> None:
    """
    Applies Bayesian optimization (single-threaded).
    """
def optimize_multi_threaded(settings: BayOptSettings, sample_tensor: torch.Tensor, callback: typing.Callable) -> None:
    """
    Applies Bayesian optimization (multi-threaded).
    """
def optimize_multi_threaded_blocks(
        settings: BayOptSettings, sample_tensor: torch.Tensor, block_size: int, block_offsets: torch.Tensor,
        callback: typing.Callable) -> None:
    """
    Applies Bayesian optimization (multi-threaded).
    """

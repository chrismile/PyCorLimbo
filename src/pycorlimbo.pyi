from __future__ import annotations
import torch
import pycorlimbo
import typing

__all__ = [
    "optimize_single_threaded",
    "optimize_multi_threaded",
    "pearson_correlation",
    "spearman_rank_correlation",
    "kendall_rank_correlation",
    "mutual_information_binned",
    "mutual_information_kraskov"
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
def pearson_correlation(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Computes the Pearson correlation coefficient of the Torch tensors X and Y.
    """
def spearman_rank_correlation(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Computes the Spearman rank correlation coefficient of the Torch tensors X and Y.
    """
def kendall_rank_correlation(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kendall rank correlation coefficient of the Torch tensors X and Y.
    """
def mutual_information_binned(
        X: torch.Tensor, Y: torch.Tensor, num_bins: int,
        X_min: float, X_max: float, Y_min: float, Y_max: float) -> torch.Tensor:
    """
    Computes the mutual information of the Torch tensors X and Y using a binning estimator.
    """
def mutual_information_kraskov(X: torch.Tensor, Y: torch.Tensor, k: int) -> torch.Tensor:
    """
    Computes the mutual information of the Torch tensors X and Y using the Kraskov estimator.
    """

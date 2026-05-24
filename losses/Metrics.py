import torch
import torch.nn.functional as F
from munch import Munch
from omegaconf import DictConfig
from .TwinsBarlow import BarlowTwinsLossHeadless, BarlowTwinsLoss

def cosine_sim_negative(*args, **kwargs):
    """
    Compute negative cosine similarity.

    Args:
        *args: Input arguments for cosine similarity.
        **kwargs: Additional keyword arguments for cosine similarity.

    Returns:
        torch.Tensor: Negative cosine similarity.
    """
    return (1. - F.cosine_similarity(*args, **kwargs)).mean()

def metric_from_str(metric, **kwargs):
    """
    Get a metric function from a string representation.

    Args:
        metric (str): Metric name.
        **kwargs: Additional keyword arguments for specific metrics.

    Returns:
        function: Metric function.
        
    Raises:
        ValueError: If the metric is invalid.
    """
    if metric == "cosine_similarity":
        return cosine_sim_negative
    elif metric in ["l1", "l1_loss", "mae"]:
        return torch.nn.functional.l1_loss
    elif metric in ["mse", "mse_loss", "l2", "l2_loss"]:
        return torch.nn.functional.mse_loss
    elif metric == "barlow_twins_headless":
        return BarlowTwinsLossHeadless(**kwargs)
    elif metric == "barlow_twins":
        return BarlowTwinsLoss(**kwargs)
    else:
        raise ValueError(f"Invalid metric for deep feature loss: {metric}")

def metric_from_cfg(metric):
    """
    Get a metric function from a configuration object.

    Args:
        metric (DictConfig): Metric configuration.

    Returns:
        function: Metric function.
        
    Raises:
        ValueError: If the metric type is invalid.
    """
    if metric.type == "cosine_similarity":
        return cosine_sim_negative
    elif metric.type in ["l1", "l1_loss", "mae"]:
        return torch.nn.functional.l1_loss
    elif metric.type in ["mse", "mse_loss", "l2", "l2_loss"]:
        return torch.nn.functional.mse_loss
    elif metric.type == "barlow_twins_headless":
        return BarlowTwinsLossHeadless(metric.feature_size)
    elif metric.type == "barlow_twins":
        layer_sizes = metric.layer_sizes if 'layer_sizes' in metric.keys() else None
        return BarlowTwinsLoss(metric.feature_size, layer_sizes)
    else:
        raise ValueError(f"Invalid metric for deep feature loss: {metric}")

def get_metric(metric):
    """
    Get a metric function based on the input representation.

    Args:
        metric (str, DictConfig, Munch, dict): Metric representation.

    Returns:
        function: Metric function.
        
    Raises:
        ValueError: If the metric representation is invalid.
    """
    if isinstance(metric, str):
        return metric_from_str(metric)
    if isinstance(metric, (DictConfig, Munch)):
        return metric_from_cfg(metric)
    if isinstance(metric, dict):
        return metric_from_cfg(Munch(metric))
    raise ValueError(f"Invalid type for metric: {type(metric)}")

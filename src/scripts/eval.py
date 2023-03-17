from typing import Dict, Callable, Iterable

import src.config as cfg
from src.utils.metrics.classification import default_metrics


def eval_model(
    model: Callable, dataloader: Iterable,
    metrics: Dict[str, Callable[[float, float], float]] = None
) -> Dict[str, float]:
    '''Compute evaluation metrics for a given model and dataset

    Parameters:
    -----------
    model : callable
        Callable which supports y_pred = `model(x)`
    dataloader : torch.utils.data.DataLoader
        Iterable with input / label pairs `(x, y)` compatible with model
    metrics : dict[str, callable]
        Dictionary of metric names and functions `fn(y_pred, y)` -> metric.

    Returns
    -------
    dict[str, float]
        Dictionary of metric names and metric values
    '''

    if metrics is None:
        metrics = default_metrics

    total_samples = 0
    metric_tallies = {name: 0 for name in metrics}

    for (x, y) in dataloader:
        x = x.to(cfg.DEVICE)
        y = y.to(cfg.DEVICE)

        y_pred = model(x)

        # Metrics
        total_samples += y.shape[0]
        for metric_name, metric_fn in metrics.items():
            metric_tallies[metric_name] += y.shape[0] * metric_fn(y_pred, y)

    metric_scores = {
        name: tally / total_samples for name, tally in metric_tallies.items()
    }

    return metric_scores

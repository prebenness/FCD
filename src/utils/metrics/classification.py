'''
Compute classification performance metrics for a single batch of data
'''

import torch


def accuracy(y_pred, y_true):
    r'''Computes k-class classification accuracy of a batch

    Parameters
    ----------
    y_pred : tensor
        (N, C) tensor of class probabilities
    y_true : tensor
        (N, 1) tensor of ordinal class labels or 
        (N, C) tensor of one-hot labels

    Returns
    -------
    float
        Classification accuracy.
    '''

    # If onehot labels, convert to ordinal
    if y_true.shape == y_pred.shape:
        y_true = torch.argmax(y_true, dim=-1)

    num_total = y_pred.shape[0]
    num_correct = torch.sum(torch.argmax(y_pred, dim=-1) == y_true)

    return (num_correct / num_total).item()


default_metrics = {
    'Accuracy': accuracy
}

'''
Test correctness of classification metrics
'''

import pytest
import torch

from src.utils.metrics.classification import accuracy


@pytest.fixture  # pylint: disable=redefined-outer-name
def example_batches():
    '''
    Return same batch of predictions and labels, first with ordinal labels
    then with one-hot encoded labels
    '''
    y_pred = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    y_true_ord = torch.tensor([2, 0])
    y_true_onehot = torch.tensor([[0, 0, 1], [1, 0, 0]])

    b1 = (y_pred, y_true_ord)
    b2 = (y_pred, y_true_onehot)

    return b1, b2


def test_accuracy(example_batches):
    '''
    Check that classification accuracy is correct and supports both ordinal 
    and one-hot labels
    '''

    b1, b2 = example_batches

    # Check correct calculation
    acc_b1 = accuracy(*b1)
    assert acc_b1 == 0.5

    # Should accept one-hot or ordinal values
    acc_b2 = accuracy(*b2)
    assert acc_b2 == acc_b1

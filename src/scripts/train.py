'''
Scripts for training a given model on a given dataset
'''

import torch
from torch import nn

import src.config as cfg
from src.utils.metrics.classification import accuracy
from src.scripts.eval import eval_model


def train_model(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters())

    for epoch_idx in range(cfg.NUM_EPOCHS):
        model = train_one_epoch(
            epoch_idx=epoch_idx, model=model, train_loader=train_loader,
            val_loader=val_loader, optimizer=optimizer
        )

    return model


def train_one_epoch(epoch_idx, model, train_loader, val_loader, optimizer):
    # Init losses
    c_loss, s_loss = 0, 0

    # Init loss functions
    c_loss_fn = nn.CrossEntropyLoss()
    s_loss_fn = nn.MSELoss()

    # Init metrics
    total_samples = 0
    acc_tally = 0

    for (x, y) in train_loader:
        # Load data to device
        x = x.to(cfg.DEVICE)
        y = y.to(cfg.DEVICE)

        # Zero gradient for every batch
        optimizer.zero_grad()

        # Forward pass
        y_pred, x_rec = model(x)

        # Compute losses
        c_loss = c_loss_fn(y_pred, y)
        s_loss = s_loss_fn(x_rec, x)

        # Tally losses
        total_loss = c_loss + s_loss

        # Backward pass
        total_loss.backward()

        # Step optimizer
        optimizer.step()

        # Update metrics
        acc_tally += y.shape[-1] * accuracy(y_pred=y_pred, y_true=y)
        total_samples += y.shape[-1]

    train_acc = acc_tally / total_samples

    # Validation metrics
    c_metrics = eval_model(
        model=lambda x: model(x)[0], dataloader=val_loader,
        metrics={
            'Accuracy': accuracy,
            'XE': c_loss_fn,
        }
    )
    s_metrics = eval_model(
        model=lambda x: model(x)[1], dataloader=val_loader,
        metrics={
            'MSE': s_loss_fn
        }
    )

    print(
        f'Epoch {epoch_idx+1:3d}: c_loss: {c_loss:.3f} s_loss: {s_loss:.3f} Acc: {train_acc:.3f}'
    )

    return model

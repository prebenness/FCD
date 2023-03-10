'''
Util functions related to fetching and processing datasets
'''

from torch.utils.data import random_split, DataLoader
import torchvision

import src.config as cfg


def get_dataloader(dataset_name, train=True):
    '''
    Returns one of the supported torchvision datasets, as a dataloader
    '''
    if dataset_name.upper() == 'MNIST':
        Dataset = torchvision.datasets.MNIST
    elif dataset_name.upper() == 'CIFAR10':
        Dataset = torchvision.datasets.CIFAR10
    elif dataset_name.upper() == 'CIFAR100':
        Dataset = torchvision.datasets.CIFAR100
    else:
        raise NotImplementedError(f'Dataset {dataset_name} not supported')

    # TODO: implement support for more transforms
    transform = torchvision.transforms.ToTensor()
    dataset = Dataset(
        'data', train=train, download=True, transform=transform
    )

    if train:
        # Split train to train, val
        train_dataset, val_dataset = random_split(dataset, [4/5, 1/5])
        train_dataloader = DataLoader(
            train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True
        )

        return train_dataloader, val_dataloader

    # Don't split test set
    test_dataloader = DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=True
    )

    return test_dataloader


def get_train_val_test_dataloader(dataset_name):
    '''
    Convenience function for getting train, val, test loaders in one go
    '''
    train_loader, eval_loader = get_dataloader(dataset_name, train=True)
    test_loader = get_dataloader(dataset_name, train=False)

    return train_loader, eval_loader, test_loader

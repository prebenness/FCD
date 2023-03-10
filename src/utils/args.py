'''
Util functions for parsing and handling args, and writing to global config
'''

import argparse

import src.config as cfg


def parse_args():
    parser = argparse.ArgumentParser(
        prog='FCD', usage='python main.py -m train'
    )
    # General args
    parser.add_argument(
        '--mode', '-m', choices=['train'], required=True
    )
    parser.add_argument(
        '--dataset', '-d', choices=['mnist', 'cifar10', 'cifar100'],
        type=lowercase_str, required=True
    )
    parser.add_argument('--cpu', action='store_true')

    # Train args
    parser.add_argument('--batch_size', required=False, type=int)

    args = parser.parse_args()
    args = sanitize_args(args)
    handle_args(args)

    return args


def sanitize_args(args):

    return args


def handle_args(args):
    set_dataset_specs(args)

    cfg.DEVICE = 'cuda' if not args.cpu else 'cpu'

    return args


def set_dataset_specs(args):
    if args.dataset == 'mnist':
        cfg.DATASET = 'mnist'
        cfg.HEIGHT, cfg.WIDTH = 28, 28
        cfg.NUM_CHANNELS = 1
        cfg.NUM_CLASSES = 10
    elif args.dataset == 'cifar10':
        cfg.DATASET = 'cifar10'
        cfg.HEIGHT, cfg.WIDTH = 32, 32
        cfg.NUM_CHANNELS = 3
        cfg.NUM_CLASSES = 10
    elif args.dataset == 'cifar100':
        cfg.DATASET = 'cifar100'
        cfg.HEIGHT, cfg.WIDTH = 32, 32
        cfg.NUM_CHANNELS = 3
        cfg.NUM_CLASSES = 100
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not supported')


def lowercase_str(s):
    s = str(s)
    return s.lower()

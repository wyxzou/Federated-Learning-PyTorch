#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import heapq
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    home = "/home/wyzou/Federated-Learning-PyTorch"

    if args.dataset == 'cifar':
        data_dir = home + '/data/cifar/'
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=transform_train)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=transform_test)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = home + '/data/mnist/'

            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

            train_dataset = datasets.MNIST(data_dir, train=True, download=False,
                                        transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=False,
                                        transform=apply_transform)
        else:
            data_dir = home + '/data/fmnist/'

            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=False,
                                        transform=apply_transform)

            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=False,
                                        transform=apply_transform)


        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def topk_weights(w, threshold):
    """
    Returns topk weights.
    """
    sparse_w = copy.deepcopy(w)

    for key in sparse_w.keys():
        # print("data type: ", sparse_w[key].dtype)
        mask = (sparse_w[key] > threshold).float()
        sparse_w[key] = torch.mul(mask, sparse_w[key])

    return sparse_w


def subtract_weights(w1, w2):
    """
    Returns topk weights.
    """
    sum_w = copy.deepcopy(w1)

    for key in sum_w.keys():
        if key in w2:
            sum_w[key] -= w2[key]

    return sum_w


def add_weights(w1, w2):
    """
    Returns topk weights.
    """
    sum_w = copy.deepcopy(w1)

    for key in sum_w.keys():
        if key in w2:
            sum_w[key] += w2[key]

    return sum_w

def initialize_memory(w):
    """
    Returns topk weights.
    """
    mem = copy.deepcopy(w)

    for key in mem.keys():
        mem[key] = 0

    return mem


def get_weight_dimension(w):
    dim = 0
    for key in w.keys():
        dim += torch.numel(w[key])

    return dim


def get_topk_value(w, k):
    w_copy = copy.deepcopy(w)

    tensors = []
    for key in w_copy.keys():
        tensors.append(torch.flatten(w_copy[key]))    

    combined_tensor = torch.cat(tensors)

    topk_val = torch.topk(combined_tensor, k)[0]

    return topk_val[len(topk_val) - 1].item()


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, \
    subtract_weights, topk_weights, add_weights, initialize_memory, get_topk_value, get_weight_dimension
from sparsification import sparsetopSGD

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)
    
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    previous_weights = copy.deepcopy(global_weights)
    delta_memory = initialize_memory(global_weights)
    gradient_size = get_weight_dimension(global_weights)
    # print("gs: ", gradient_size)
    k = int(gradient_size * args.topk)

    m = max(int(args.frac * args.num_users), 1)
    
    # print("bidirectional: ", args.bidirectional)
    local_models = []
    optimizers = []

    for i in range(m):
        local_model = copy.deepcopy(global_model)
        local_models.append(local_model)
        if args.optimizer == 'sparsetopk':
            optimizer = sparsetopSGD(local_model.parameters(), lr=args.lr, topk=args.topk)
            optimizers.append(optimizer)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(local_model.parameters(), lr=args.lr, momentum=0.5)
            optimizers.append(optimizer)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=1e-4)
            optimizers.append(optimizer)


    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        for i in range(m):
            local_models[i].load_state_dict(global_weights)

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # print("idxs_users", idxs_users)
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            # w, loss = local_model.update_weights_with_memory(
            #    model=copy.deepcopy(global_model), global_round=epoch, optimizer=optimizers[idx])
            w, loss = local_model.update_weights_with_memory(
                model=local_models[idx], global_round=epoch, optimizer=optimizers[idx])
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))


        if args.bidirectional == 1 and args.optimizer == 'sparsetopk':
            # update global weights
            global_weights = average_weights(local_weights)

            # print("global weights: ", global_weights.keys())
            g = subtract_weights(previous_weights, global_weights)
            corrected_weights = add_weights(g, delta_memory)

            threshold = get_topk_value(corrected_weights, k)
            # print("threshold", threshold)

            topk = topk_weights(corrected_weights, threshold)

            # print("corrected weights: ", corrected_weights.keys())
            # print("topk weights: ", topk.keys())

            delta_memory = subtract_weights(corrected_weights, topk)
                
            global_weights = subtract_weights(previous_weights, topk)


        # update global weights
        global_weights = average_weights(local_weights)
        previous_weights = copy.deepcopy(global_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = 'save/objects/{}_{}_EPOCH[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_DIR[{}]_NUM[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_bs, args.optimizer, args.bidirectional, args.number)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))

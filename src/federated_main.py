#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import pdb

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
import models
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, VGG, MLPFashion_Mnist
from utils import get_dataset, average_weights, exp_details, update_non_keys, \
    subtract_values, topk_values, add_values, initialize_memory, get_topk_value, get_weight_dimension, count_nonzero_dict, get_gradient_from_dict
from sparsification import sparsetopSGD


# Batch normalization
# https://kaixih.github.io/batch-norm/
# https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/#:~:text=Batch%20normalization%20is%20a%20technique,required%20to%20train%20deep%20networks.
# https://stats.stackexchange.com/questions/219808/how-and-why-does-batch-normalization-use-moving-averages-to-track-the-accuracy-o

# https://edgify.medium.com/distributed-training-on-edge-devices-batch-normalization-with-non-iid-data-e73ca9146260
# https://discuss.pytorch.org/t/how-does-batchnorm-keeps-track-of-running-mean/40084
# https://discuss.pytorch.org/t/implementing-batchnorm-in-pytorch-problem-with-updating-self-running-mean-and-self-running-var/49314
# https://discuss.pytorch.org/t/updating-running-mean-and-running-var-in-a-custom-batchnorm/77464


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
        if args.dataset == 'mnist':
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
        
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                                dim_out=args.num_classes)
        elif args.dataset == 'fmnist':
            global_model = MLPFashion_Mnist()
    elif args.model == 'vgg':
        global_model = VGG('VGG19')
    else:
        exit('Error: unrecognized model')

    home = "/home/wyzou/Federated-Learning-PyTorch"

    start_epoch = 0
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(home + '/vgg/checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(home + '/vgg/checkpoint/dummy.pth')
        global_model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']


    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    print("global weight keys: ", global_weights.keys())

    update_weight_keys = set()
    for name, param in global_model.named_parameters():
        if param.requires_grad:
            update_weight_keys.add(name)

    print("weights to update: ", global_weights.keys())
    # Training
    train_loss, test_losses, valid_accuracy, test_accuracies = [], [], [], []
    xi_values, delta_upstream_values, delta_downstream_values = [], [], []
    information_compression_upstream, information_compression_downstream = [], []


    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    previous_weights = copy.deepcopy(global_weights)
    delta_memory = initialize_memory(global_weights)
    gradient_size = get_weight_dimension(global_weights, update_weight_keys)
    print("gradient size: ", gradient_size)
    k = int(gradient_size * args.topk)
    k_d = int(gradient_size * args.topk_d)

    m = max(int(args.frac * args.num_users), 1)
    
    local_models = []
    optimizers = []
    schedulers = []

    for i in range(args.num_users):
        local_model = copy.deepcopy(global_model)
        local_models.append(local_model)
        if args.optimizer == 'sparsetopk':
            optimizer = sparsetopSGD(local_model.parameters(), lr=args.lr, topk=args.topk)
            optimizers.append(optimizer)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(local_model.parameters(), lr=args.lr, momentum=0)
            optimizers.append(optimizer)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=1e-4)
            optimizers.append(optimizer)


    # https://stackoverflow.com/questions/54746829/pytorch-whats-the-difference-between-state-dict-and-parameters
    print("Completed model initialization")

    for epoch in tqdm(range(args.epochs)):
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        batch_number = 0
        local_updates = {}
        worker_batch_size = []
        print("Starting to initialize local updates")

        batch_xi_values = []

        batch_upstream_compression_values, batch_downstream_compression_values = [], []
        batch_delta_upstream_values, batch_delta_downstream_values  = [], []

        for idx in idxs_users:
            local_update = LocalUpdate(args=args, dataset=train_dataset,
                                       idxs=user_groups[idx], logger=logger)
            local_updates[idx] = local_update
            worker_batch_size.append(local_update.batches_per_epoch())


        num_batches = min(worker_batch_size)
        
        for i in range(num_batches):
            local_weights = []
            local_losses = [[] for i in range(args.num_users)]

            topk_gradient_with_errors, gradient_with_errors, raw_gradients  = [], [], []

            gradient_before_topk = []
            gradient_after_topk = []
            gradient_without_error = []

            user_gradient_compression = []
            worker_to_server_errors  = []
            
            for idx in idxs_users:
                w, loss = local_updates[idx].update_weights_per_batch(
                    model=local_models[idx], global_round=epoch, optimizer=optimizers[idx], batch_idx=i)

                local_weights.append(copy.deepcopy(w))
                local_losses[idx].append(copy.deepcopy(loss))

                if args.measure_parameters == 1 and args.optimizer == "sparsetopk":
                    gradient_before_topk.append(optimizers[idx].gradient_before_topk)
                    gradient_after_topk.append(optimizers[idx].gradient_after_topk)
                    gradient_without_error.append(optimizers[idx].gradient_without_error)
                    worker_to_server_errors.append(optimizers[idx].error)

                    # calculate percent compression
                    magnitude_before_compression = torch.linalg.norm(optimizers[idx].gradient_before_topk).item()
                    compression_difference = torch.linalg.norm(optimizers[idx].gradient_after_topk - optimizers[idx].gradient_before_topk).item()

                    delta_compression = compression_difference / magnitude_before_compression
                    user_gradient_compression.append(delta_compression)


            if args.measure_parameters == 1 and args.optimizer == 'sparsetopk':
                batch_delta_upstream_values.append(max(user_gradient_compression))

                # topk applied before aggregation, this is sum(topk(g + e))
                topk_before_aggregation = torch.stack(gradient_after_topk, dim=0).sum(dim=0)
                # topk applied after aggregation, sum(g + e)
                topk_after_aggregation = torch.stack(gradient_before_topk, dim=0).sum(dim=0)

                error_sum = torch.stack(worker_to_server_errors, dim=0).sum(dim=0)
                error_sum = error_sum / args.num_users

                nonzero_before_downstream_compress = torch.count_nonzero(topk_before_aggregation, dim=0).item()
                batch_upstream_compression_values.append(nonzero_before_downstream_compress)

            if args.measure_parameters == 1 and args.bidirectional == 0 and args.optimizer == 'sparsetopk':
                # gradient_sum_with_errors is sum(g + e)
                gradient_sum_with_errors = topk_after_aggregation.detach().clone()


                _, indices = torch.topk(topk_after_aggregation, int((1 - args.topk) *  topk_after_aggregation.shape[0]), dim=0, largest=False)
                topk_after_aggregation[indices] = 0
                # topk_after_aggregation is now topk(sum(g + e))
                # topk(sum(g + e)) - sum(topk(g + e))
                gap = torch.linalg.norm(topk_after_aggregation - topk_before_aggregation).item()
                g = torch.linalg.norm(torch.stack(gradient_without_error, dim=0).sum(dim=0)).item()

                gap_topk_before_aggregation = torch.linalg.norm(gradient_sum_with_errors - topk_before_aggregation).item()
                gap_topk_after_aggregation = torch.linalg.norm(gradient_sum_with_errors - topk_after_aggregation).item()
                g_topk_before_aggregation = torch.linalg.norm(gradient_sum_with_errors).item()

                batch_xi_values.append(gap/g)

                adjusted_error = torch.linalg.norm(error_sum).item()

            global_weights = average_weights(local_weights)

            if args.bidirectional == 1 and args.optimizer == 'sparsetopk':
                previous_weights = update_non_keys(previous_weights, global_weights, update_weight_keys)
                g = subtract_values(previous_weights, global_weights, update_weight_keys)
                corrected_gradients = add_values(g, delta_memory, update_weight_keys)

                threshold = get_topk_value(corrected_gradients, k_d, update_weight_keys)
                topk = topk_values(corrected_gradients, threshold, update_weight_keys)

                ## optional measurements
                if args.measure_parameters == 1:

                    delta_t = get_gradient_from_dict(delta_memory, update_weight_keys)
                    
                    topk_after_aggregation = delta_t + torch.div(topk_after_aggregation, args.num_users)
                    gradient_sum_with_errors = topk_after_aggregation.detach().clone()

                    _, indices = torch.topk(topk_after_aggregation, int((1 - args.topk_d) *  topk_after_aggregation.shape[0]), dim=0, largest=False)
                    topk_after_aggregation[indices] = 0


                    topk_before_aggregation = delta_t + torch.div(topk_before_aggregation, args.num_users)
                    # clone is used to measure gradient sparsity before downstream compression
                    topk_before_aggregation_clone = topk_before_aggregation.clone()
                    magnitude_before_compression = torch.linalg.norm(topk_before_aggregation).item()

                    _, indices = torch.topk(topk_before_aggregation, int((1 - args.topk_d) *  topk_before_aggregation.shape[0]), dim=0, largest=False)
                    topk_before_aggregation[indices] = 0

                    compression_difference = torch.linalg.norm(topk_before_aggregation - topk_before_aggregation_clone).item()

                    delta_compression = compression_difference / magnitude_before_compression
                    batch_delta_downstream_values.append(delta_compression)

                    gap_topk_before_aggregation = torch.linalg.norm(gradient_sum_with_errors - topk_before_aggregation).item()
                    gap_topk_after_aggregation = torch.linalg.norm(gradient_sum_with_errors - topk_after_aggregation).item()
                    g_topk_before_aggregation = torch.linalg.norm(gradient_sum_with_errors).item()

                    gap = torch.linalg.norm(topk_after_aggregation - topk_before_aggregation).item()
                    g = torch.linalg.norm(torch.stack(gradient_without_error, dim=0).sum(dim=0)).item()

                    batch_xi_values.append(gap/g)

                    nonzero_after_downstream_compress = count_nonzero_dict(topk, update_weight_keys)
                    batch_downstream_compression_values.append(nonzero_after_downstream_compress/nonzero_before_downstream_compress)

                ## optional measurements end
                delta_memory = subtract_values(corrected_gradients, topk, update_weight_keys)
                
                # This has the original global weight running mean, running var
                global_weights = subtract_values(previous_weights, topk, update_weight_keys)

                # update global weights
                previous_weights = copy.deepcopy(global_weights)

                
            # update global weights
            global_model.load_state_dict(global_weights)
            for worker in range(args.num_users):
                local_models[worker].load_state_dict(global_weights)


        if args.measure_parameters == 1 and args.optimizer == 'sparsetopk':
            xi_values.extend(batch_xi_values)

            information_compression_upstream.extend(batch_upstream_compression_values)
            information_compression_downstream.extend(batch_downstream_compression_values)
            delta_upstream_values.extend(batch_delta_upstream_values)
            delta_downstream_values.extend(batch_delta_downstream_values)


        loss_avgs = []
        for local_loss in local_losses:
            loss_avgs.append( sum(local_loss)/len(local_losses) )


        loss_avg = sum(loss_avgs) / len(loss_avgs)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
            
        global_model.eval()
        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test_dataset)

        test_accuracies.append(test_acc)
        test_losses.append(test_loss)

        if args.validation == 1:
            for c in range(args.num_users):
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx], logger=logger)
                acc, loss = local_model.inference(model=global_model)
                list_acc.append(acc)
                list_loss.append(loss)
            
            valid_accuracy.append(sum(list_acc)/len(list_acc))


        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            if args.validation == 1:
                print('Validation Accuracy: {:.2f}% \n'.format(100*valid_accuracy[-1]))
            print("Test Accuracy: {:.2f}%".format(100*test_acc))


        # Save checkpoint
        if args.save == 1:
            print('Saving..')
            state = {
                'net': global_model.state_dict(),
                'acc': test_acc,
                'epoch': start_epoch + epoch + 1,
            }
            torch.save(state, home + '/vgg/checkpoint/ckpt_DATASET[{}]_LR[{}]_DIR[{}]_EPOCH[{}].pth'.format(args.dataset, args.lr, args.bidirectional, start_epoch + args.epochs))


    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')

    if args.validation == 1:
        print("|---- Avg Validation Accuracy: {:.2f}%".format(100*valid_accuracy[-1]))

    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = home + '/save/{}-{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl'.\
        format(args.dataset, args.model, args.dataset, args.model, args.epochs, args.num_users, args.frac, args.iid,
               args.local_bs, args.optimizer, args.lr, args.bidirectional, args.topk, args.topk_d, args.number)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, valid_accuracy, test_losses, test_accuracies, xi_values, information_compression_upstream, information_compression_downstream, delta_upstream_values, delta_downstream_values], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))


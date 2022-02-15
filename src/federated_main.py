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
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(home + '/vgg/checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(home + '/vgg/checkpoint/ckpt.pth')
        global_model.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        # Set the model to train and send it to device.


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
    train_loss, test_losses, train_accuracy, test_accuracies = [], [], [], []
    xi_values, delta_upstream_values, delta_downstream_values = [], [], []
    information_compression_upstream, information_compression_downstream = [], []

    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    previous_weights = copy.deepcopy(global_weights)
    delta_memory = initialize_memory(global_weights)
    gradient_size = get_weight_dimension(global_weights, update_weight_keys)
    # print("gradient size: ", gradient_size)
    k = int(gradient_size * args.topk)
    k_d = int(gradient_size * args.topk_d)

    m = max(int(args.frac * args.num_users), 1)
    
    # print("bidirectional: ", args.bidirectional)
    local_models = []
    optimizers = []
    schedulers = []


    for i in range(args.num_users):
        local_model = copy.deepcopy(global_model)
        local_models.append(local_model)
        if args.optimizer == 'sparsetopk':
            optimizer = sparsetopSGD(local_model.parameters(), lr=args.lr, topk=args.topk)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, )
            optimizers.append(optimizer)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(local_model.parameters(), lr=args.lr, momentum=0)
            optimizers.append(optimizer)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=1e-4)
            optimizers.append(optimizer)


    # https://stackoverflow.com/questions/54746829/pytorch-whats-the-difference-between-state-dict-and-parameters
    print("Completed model initialization")

    # pdb.set_trace()
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
        # for worker in range(args.num_users):
            local_update = LocalUpdate(args=args, dataset=train_dataset,
                                       idxs=user_groups[idx], logger=logger)
            # print("idx and user group length: ", idx, len(user_groups[idx]))
            local_updates[idx] = local_update
            worker_batch_size.append(local_update.batches_per_epoch())

    
        # print("Finished local updates")
        print("idxs_users", idxs_users)
        print("worker batch size: ", worker_batch_size)

        num_batches = min(worker_batch_size)
        
        # print("number of batches: ", num_batches)


        print("Starting batches in epoch")
        for i in range(num_batches):
            local_weights, local_losses = [], []
            topk_gradient_with_errors, gradient_with_errors, raw_gradients  = [], [], []
            for worker in range(args.num_users):
                local_models[worker].load_state_dict(global_weights)

            gradient_before_topk = []
            gradient_after_topk = []
            gradient_without_error = []

            user_gradient_compression = []
            
            for idx in idxs_users:
                # local_model = LocalUpdate(args=args, dataset=train_dataset,
                #                          idxs=user_groups[idx], logger=logger)
                w, loss = local_updates[idx].update_weights_per_batch(
                    model=local_models[idx], global_round=epoch, optimizer=optimizers[idx], batch_idx=i)
                # w, loss = local_model.update_weights_with_memory(
                #    model=local_models[idx], global_round=epoch, optimizer=optimizers[idx])

                # topk_gradient_with_errors.append(optimizers[idx].topk_gradient_with_error)
                # gradient_with_errors.append(optimizers[idx].gradient_with_error)
                # raw_gradients.append(optimizers[idx].raw_gradient)

                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
                # print("idx: ", idx, "processed labels", local_model.check_mnist_labels()

                if args.measure_parameters == 1 and args.optimizer == "sparsetopk":
                    # information_compression.append(torch.linalg.norm(optimizers[idx].gradient_after_topk).item() / torch.linalg.norm(optimizers[idx].gradient_before_topk).item())

                    gradient_before_topk.append(optimizers[idx].gradient_before_topk)
                    gradient_after_topk.append(optimizers[idx].gradient_after_topk)
                    gradient_without_error.append(optimizers[idx].gradient_without_error)

                    # calculate percent compression
                    magnitude_before_compression = torch.linalg.norm(optimizers[idx].gradient_before_topk).item()
                    compression_difference = torch.linalg.norm(optimizers[idx].gradient_after_topk - optimizers[idx].gradient_before_topk).item()

                    delta_compression = compression_difference / magnitude_before_compression
                    user_gradient_compression.append(delta_compression)


            if args.measure_parameters == 1 and args.optimizer == 'sparsetopk':
                batch_delta_upstream_values.append(max(user_gradient_compression))

                # topk applied before aggregation
                topk_before_aggregation = torch.stack(gradient_after_topk, dim=0).sum(dim=0)
                # topk applied after aggregation
                topk_after_aggregation = torch.stack(gradient_before_topk, dim=0).sum(dim=0)

                nonzero_before_downstream_compress = torch.count_nonzero(topk_before_aggregation, dim=0).item()


            if args.measure_parameters == 1 and args.bidirectional == 0 and args.optimizer == 'sparsetopk':
                _, indices = torch.topk(topk_after_aggregation, int((1 - args.topk) *  topk_after_aggregation.shape[0]), dim=0, largest=False)
                topk_after_aggregation[indices] = 0
                
                gap = torch.linalg.norm(topk_after_aggregation - topk_before_aggregation).item()
                g = torch.linalg.norm(torch.stack(gradient_without_error, dim=0).sum(dim=0)).item()

                batch_xi_values.append(gap/g)

                batch_upstream_compression_values.append(nonzero_before_downstream_compress)


            global_weights = average_weights(local_weights)


            if args.bidirectional == 1 and args.optimizer == 'sparsetopk':
                # pdb.set_trace()
                previous_weights = update_non_keys(previous_weights, global_weights, update_weight_keys)
                g = subtract_values(previous_weights, global_weights, update_weight_keys)
                # pdb.set_trace()
                corrected_gradients = add_values(g, delta_memory, update_weight_keys)

                threshold = get_topk_value(corrected_gradients, k_d, update_weight_keys)
                topk = topk_values(corrected_gradients, threshold, update_weight_keys)

                ## optional measurements
                if args.measure_parameters == 1:
                    delta_t = get_gradient_from_dict(delta_memory, update_weight_keys)
                    topk_after_aggregation = delta_t + torch.div(topk_after_aggregation, args.num_users)
                    _, indices = torch.topk(topk_after_aggregation, int((1 - args.topk_d) *  topk_after_aggregation.shape[0]), dim=0, largest=False)
                    topk_after_aggregation[indices] = 0



                    topk_before_aggregation = delta_t + torch.div(topk_before_aggregation, args.num_users)
                    topk_before_aggregation_clone = topk_before_aggregation.clone()
                    magnitude_before_compression = torch.linalg.norm(topk_before_aggregation).item()

                    _, indices = torch.topk(topk_before_aggregation, int((1 - args.topk_d) *  topk_before_aggregation.shape[0]), dim=0, largest=False)
                    topk_before_aggregation[indices] = 0

                    compression_difference = torch.linalg.norm(topk_before_aggregation - topk_before_aggregation_clone).item()

                    delta_compression = compression_difference / magnitude_before_compression
                    batch_delta_downstream_values.append(delta_compression)



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


        # for idx in idxs_users:
        #    print("idx: ", idx, "processed labels", local_updates[idx].check_mnist_labels())
        if args.measure_parameters == 1 and args.optimizer == 'sparsetopk':
            xi_values.append(max(batch_xi_values, default=-1))
            information_compression_upstream.append(max(batch_upstream_compression_values, default=-1))
            information_compression_downstream.append(max(batch_downstream_compression_values, default=-1))
            delta_upstream_values.append(max(batch_delta_upstream_values, default=-1))
            delta_downstream_values.append(max(batch_delta_downstream_values, default=-1))

        # xi_values.append(max(batch_xi_values))
        # information_compression.append(max(batch_compression_values))

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
            
        global_model.eval()
        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test_dataset)

        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)

        test_accuracies.append(test_acc)
        test_losses.append(test_loss)

        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            print("Test Accuracy: {:.2f}%".format(100*test_acc))


        # Save checkpoint
        print('Saving..')
        state = {
            'net': global_model.state_dict(),
            'acc': test_acc,
            'epoch': start_epoch + args.epochs,
        }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        torch.save(state, home + '/vgg/checkpoint/ckpt_LR[{}]_DIR[{}]_EPOCH[{}].pth'.format(args.lr, args.bidirectional, start_epoch + args.epochs))
        best_acc = acc


    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = home + '/save/{}/{}_{}_EPOCH[{}]_USERS[{}]_C[{}]_iid[{}]_B[{}]_OPT[{}]_LR[{}]_DIR[{}]_TOPK[{}]_TOPKD[{}]_NUM[{}].pkl'.\
        format(args.dataset, args.dataset, args.model, args.epochs, args.num_users, args.frac, args.iid,
               args.local_bs, args.optimizer, args.lr, args.bidirectional, args.topk, args.topk_d, args.number)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy, test_losses, test_accuracies, xi_values, information_compression_upstream, information_compression_downstream, delta_upstream_values ,delta_downstream_values], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))




    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r') 339 637asdf
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

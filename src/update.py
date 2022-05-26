#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # image_tensor = torch.tensor(image)
        # label_tensor = torch.tensor(label)
        # print("image type: ", image_tensor.dtype)
        # print("label type: ", label_tensor.dtype)
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        # print("train val test started")
        self.trainloader, self.validloader, self.trainindices, self.validindices = self.train_val_test(
            dataset, list(idxs))
        # print("train val test complete")
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        
        if self.args.model == 'vgg': 
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        else:
            self.criterion = nn.NLLLoss().to(self.device)

        self.num_labels = [0] * 10
        self.iterable_trainloader = iter(self.trainloader)

    # def check_mnist_labels(self):
    #    return self.num_labels

    def batches_per_epoch(self):
        return math.floor(len(self.trainloader.dataset) / self.args.local_bs)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        # print("Starting train val test")
        if self.args.validation == 1:
            idxs_train = idxs[:int(0.8*len(idxs))]
        else:
            idxs_train = idxs[:len(idxs)]
        idxs_val = idxs[int(0.8*len(idxs)):]
        # idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        # testloader = DataLoader(DatasetSplit(dataset, idxs_test),
        #                        batch_size=int(len(idxs_test)/10), shuffle=False)
        # print("Data Loader complete")
        return trainloader, validloader, idxs_train, idxs_val

    # https://stackoverflow.com/questions/54053868/how-do-i-get-a-loss-per-epoch-and-not-per-batch
    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        # for iter in range(self.args.local_ep):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            model.zero_grad()
            log_probs = model(images)
            loss = self.criterion(log_probs, labels)
            loss.backward()
            optimizer.step()

            if self.args.verbose and (batch_idx % 10 == 0):
                print('Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    global_round, batch_idx * len(images),
                    len(self.trainloader.dataset),
                    100. * batch_idx / len(self.trainloader), loss.item()))
            self.logger.add_scalar('loss', loss.item())
            batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def update_weights_with_memory(self, model, global_round, optimizer):
        model.train()

        epoch_loss = []
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(self.trainloader):
            
            # record labels for analysis
            # batch_labels = labels.numpy()
            # for l in batch_labels:
            #    self.num_labels[int(l)] += 1

            images, labels = images.to(self.device), labels.to(self.device)

            
            model.zero_grad()
            log_probs = model(images)
            loss = self.criterion(log_probs, labels)
            loss.backward()
            optimizer.step()

            if self.args.verbose and (batch_idx % 10 == 0):
                print('Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    global_round, batch_idx * len(images),
                    len(self.trainloader.dataset),
                    100. * batch_idx / len(self.trainloader), loss.item()))
            self.logger.add_scalar('loss', loss.item())
            batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def update_weights_per_batch(self, model, global_round, optimizer, batch_idx):
        model.train()

        # epoch_loss = []
        # batch_loss = []

        images, labels = next(self.iterable_trainloader)
        # print("label: ", labels)

        # batch_labels = labels.numpy()

        # for l in batch_labels:
        #    self.num_labels[int(l)] += 1

        images, labels = images.to(self.device), labels.to(self.device)
        
        model.zero_grad()
        log_probs = model(images)
        loss = self.criterion(log_probs, labels)
        # print("log_probs type in criterion: ", log_probs.dtype)
        # print("label type in criteriont: ", labels.dtype)

        loss.backward()
        optimizer.step()

        # if self.args.verbose and (batch_idx % 10 == 0):
        #     print('Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         global_round, batch_idx * len(images),
        #         len(self.trainloader.dataset),
        #         100. * batch_idx / len(self.trainloader), loss.item()))
        self.logger.add_scalar('loss', loss.item())
        # batch_loss.append(loss.item())
        # epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), loss.item() # sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.validloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'

    if args.model == 'vgg':
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = nn.NLLLoss().to(device)

    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels.long())
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    accuracy = correct/total
    return accuracy, loss

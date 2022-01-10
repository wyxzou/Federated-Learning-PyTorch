import os
import time
import datetime
import numpy as np
# import horovod.torch as hvd


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader, Dataset

import torch.distributed as dist
import torch.utils.data.distributed

import argparse

parser = argparse.ArgumentParser(description='mnist classification models, distributed data parallel test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=768, help='')
parser.add_argument('--max_epochs', type=int, default=4, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')
# parser.add_argument('--gpu', type=int, default=1, help='')

parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='gloo', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')


def main():
    print("Starting...")

    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()

    """ This next line is the key to getting DistributedDataParallel working on SLURM:
		SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
 		current process inside a node and is also 0 or 1 in this example."""

    local_rank = int(os.environ.get("SLURM_LOCALID")) 
    rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank

    current_device = local_rank

    torch.cuda.set_device(current_device)

    """ this block initializes a process group and initiate communications
		between all processes running on all nodes """

    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    #init the process group
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)
    print("process group ready!")

    print('From Rank: {}, ==> Making model..'.format(rank))

    class MLP(nn.Module):
        def __init__(self, dim_in=784, dim_hidden=64, dim_out=10):
            super(MLP, self).__init__()
            self.layer_input = nn.Linear(dim_in, dim_hidden)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout()
            self.layer_hidden = nn.Linear(dim_hidden, dim_out)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
            x = self.layer_input(x)
            x = self.dropout(x)
            x = self.relu(x)
            x = self.layer_hidden(x)
            return self.softmax(x)


    class Net(nn.Module):

       def __init__(self):
          super(Net, self).__init__()

          self.conv1 = nn.Conv2d(3, 6, 5)
          self.pool = nn.MaxPool2d(2, 2)
          self.conv2 = nn.Conv2d(6, 16, 5)
          self.fc1 = nn.Linear(16 * 5 * 5, 120)
          self.fc2 = nn.Linear(120, 84)
          self.fc3 = nn.Linear(84, 10)

       def forward(self, x):
          x = self.pool(F.relu(self.conv1(x)))
          x = self.pool(F.relu(self.conv2(x)))
          x = x.view(-1, 16 * 5 * 5)
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = self.fc3(x)
          return x

    net = MLP()

    net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[current_device])

    print('From Rank: {}, ==> Preparing data..'.format(rank))

    transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    data_dir = "./data/mnist/"
    dataset_train = MNIST(root=data_dir, train=True, download=False, transform=transform_train)

    # dataset_test = MNIST(root=data_dir, train=False, download=False, transform=transform_train)

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler)

    criterion = nn.NLLLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=float(args.lr), momentum=0, weight_decay=0)

    for epoch in range(args.max_epochs):

        train_sampler.set_epoch(epoch)

        train(epoch, net, criterion, optimizer, train_loader, rank)


# def horovod_main():
#     args = parser.parse_args()

#     hvd.init()

#     print("Starting...")

#     local_rank = hvd.local_rank()
#     global_rank = hvd.rank()

#     torch.cuda.set_device(local_rank)

#     class MLP(nn.Module):
#         def __init__(self, dim_in=784, dim_hidden=64, dim_out=10):
#             super(MLP, self).__init__()
#             self.layer_input = nn.Linear(dim_in, dim_hidden)
#             self.relu = nn.ReLU()
#             self.dropout = nn.Dropout()
#             self.layer_hidden = nn.Linear(dim_hidden, dim_out)
#             self.softmax = nn.Softmax(dim=1)

#         def forward(self, x):
#             x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
#             x = self.layer_input(x)
#             x = self.dropout(x)
#             x = self.relu(x)
#             x = self.layer_hidden(x)
#             return self.softmax(x)


#     class Net(nn.Module):

#        def __init__(self):
#           super(Net, self).__init__()

#           self.conv1 = nn.Conv2d(3, 6, 5)
#           self.pool = nn.MaxPool2d(2, 2)
#           self.conv2 = nn.Conv2d(6, 16, 5)
#           self.fc1 = nn.Linear(16 * 5 * 5, 120)
#           self.fc2 = nn.Linear(120, 84)
#           self.fc3 = nn.Linear(84, 10)

#        def forward(self, x):
#           x = self.pool(F.relu(self.conv1(x)))
#           x = self.pool(F.relu(self.conv2(x)))
#           x = x.view(-1, 16 * 5 * 5)
#           x = F.relu(self.fc1(x))
#           x = F.relu(self.fc2(x))
#           x = self.fc3(x)
#           return x

#     net = MLP()

#     net.cuda()
#     net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[current_device])

#     print('From Rank: {}, ==> Preparing data..'.format(rank))

#     transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

#     data_dir = "./data/mnist/"
#     dataset_train = MNIST(root=data_dir, train=True, download=False, transform=transform_train)

#     # dataset_test = MNIST(root=data_dir, train=False, download=False, transform=transform_train)

#     train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
#     train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler)

#     criterion = nn.NLLLoss().cuda()
#     optimizer = optim.SGD(net.parameters(), lr=float(args.lr), momentum=0, weight_decay=0)

#     optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=net.named_parameters())
#     hvd.broadcast_parameters(net.state_dict(), root_rank=0)

#     for epoch in range(args.max_epochs):

#         train_sampler.set_epoch(epoch)

#         train(epoch, net, criterion, optimizer, train_loader, rank)



def mpi_main():
    print("Starting...")

    args = parser.parse_args()

    rank = os.environ.get("SLURM_LOCALID")
    print("rank: ", rank)

    current_device = 0
    torch.cuda.set_device(current_device)

    """ this block initializes a process group and initiate communications
                between all processes that will run a model replica """

    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))

    dist.init_process_group(backend="mpi", init_method=args.init_method) 
    # Use backend="mpi" or "gloo". NCCL does not work on a single GPU due to a hard-coded multi-GPU topology check.

    print('From Rank: {}, ==> Making model..'.format(rank))

    class MLP(nn.Module):
        def __init__(self, dim_in=784, dim_hidden=64, dim_out=10):
            super(MLP, self).__init__()
            self.layer_input = nn.Linear(dim_in, dim_hidden)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout()
            self.layer_hidden = nn.Linear(dim_hidden, dim_out)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
            x = self.layer_input(x)
            x = self.dropout(x)
            x = self.relu(x)
            x = self.layer_hidden(x)
            return self.softmax(x)


    net = MLP()

    net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[current_device])

    print('From Rank: {}, ==> Preparing data..'.format(rank))

    transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    data_dir = "./data/mnist/"
    dataset_train = MNIST(root=data_dir, train=True, download=False, transform=transform_train)

    dataset_test = MNIST(root=data_dir, train=False, download=False, transform=transform_train)

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler)

    criterion = nn.NLLLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=float(args.lr), momentum=0, weight_decay=0)

    for epoch in range(args.max_epochs):

        train_sampler.set_epoch(epoch)

        train(epoch, net, criterion, optimizer, train_loader, rank, dataset_test)



def train(epoch, net, criterion, optimizer, train_loader, train_rank, dataset_test):
    train_loss = 0
    correct = 0
    total = 0

    test_accuracy = -1
    test_loss = -1

    epoch_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        net.train()

        start = time.time()

        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100 * correct / total

        batch_time = time.time() - start

        elapse_time = time.time() - epoch_start
        # elapse_time = datetime.timedelta(seconds=elapse_time)
        # print("From Rank: {}, Training time {}".format(train_rank, elapse_time))
    else:
        #if train_rank == 0:
        #    print("checking test accuracy...")
        test_accuracy, test_loss = test_inference(net, dataset_test)

        l = train_loss/len(train_loader)
        print("From Rank: {}, Epoch {} - Training loss: {}, Test accuracy: {}, ".format(train_rank, epoch, l, test_accuracy))



def test_inference(model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss


if __name__=='__main__':
   mpi_main()

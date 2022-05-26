# Bidirectional TopK SGD

Implementation of Bidirectional and Unidirectional TopK SGD. Original repository located at https://github.com/AshwinRJ/Federated-Learning-PyTorch.

Experiments are produced on MNIST, Fashion MNIST and CIFAR10.

MLP, CNN and VGG19 networks are tested. 

## Requirments
Install all the packages from requirments.txt

## Data
* Experiments are run on MNIST, Fashion MNIST and CIFAR10.
* Download datasets into their respective directories.

## Running the experiments

Experiments are run on a Compute Canada environment using SLURM. To run experiments, find a script in the ```fmnist-mlp```, ```fmnist-cnn```, ```mnist-mlp```, ```mnist-mlp```, ```cifar-vgg19``` folders under the root directory and run with ```sbatch ./fmnist_c1_u10_iid1_dir1.sh```. To run without SLURM, remove the SLURM environment variables and setup similar to ```run_without_slurm.sh```. We briefly explain the meaning behind ```fmnist_c1_u10_iid1_dir1.sh``` the filename: fmnist - dataset, u10 - 10 workers, iid1 - distribution of data on workers is IID, dir1 - bidirectional topk sparsification is used (dir0 means that unidirectional topk sparsification is used). Any script that configures experiments to run with the SGD optimizer will look like the following: ```cifar_vgg_sgd_c1_u50_iid1.sh```.

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Options: 'mnist', 'fmnist', 'cifar'
* ```--model:```    Options: 'mlp', 'cnn', 'vgg'
* ```--gpu:```      Default: None (runs on CPU). Can also be set to the specific gpu id
* ```--epochs:```   Number of rounds of training.
* ```--lr:```       Learning rate 
* ```--local_bs:``` Batch size
* ```--iid:```      Distribution of data amongst users. Default set to IID. Set to 0 for non-IID
* ```--measure_parameters``` 1 to measure constants in convergence bound and compression factor, 0 to ignore
* ```--validation``` 1 to use validation set, 0 to only use training and testing set
* ```--topk:``` percent of gradient size to be kept after uplink sparsification
* ```--topkd:``` percent of gradient size to be kept after downlink sparsification
* ```--users:``` number of workers in parameter server
* ```--bidirectional:``` 1 to train with bidirectional topk, 0 to train with unidirectiona topk
* ```--optimizer:``` optimizer used, can be sparsetopk or sgd

Examples can be found in ```fmnist-mlp```, ```fmnist-cnn```, ```mnist-mlp```, ```mnist-mlp```, and ```cifar-vgg19``` folders under the root directory.


## Plots

Scripts to draw plots an be found in the ```plot``` folder under the root directory


## Toy Example

The jupyter notebook with our toy example can be found in ```toy_example``` folder


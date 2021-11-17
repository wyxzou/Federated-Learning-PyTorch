#!/bin/sh
#SBATCH --account=def-desterck
#SBATCH	--nodes=1
#SBATCH	--gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=03:00:00


REPEATS=1
MODEL=mlp
IID=0
BIDIRECTIONAL=0
EPOCH=50
OPTIMIZER=sparsetopk
DATASET=mnist
FRAC=0.01
GPU=cuda:0
USERS=1
LOCAL_BS=10
LR=0.01

export SLURM_TMPDIR=/home/wyzou/Federated-Learning-PyTorch/
module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch torchvision tqdm tensorboardX six --no-index

srun python src/federated_main.py --model=$MODEL --local_ep=1 --local_bs=$LOCAL_BS --frac=$FRAC --num_users=$USERS --optimizer=$OPTIMIZER \
	 --bidirectional=$BIDIRECTIONAL --dataset=$DATASET --gpu=$GPU --lr=$LR --iid=$IID --epochs=$EPOCH --number=$idx;

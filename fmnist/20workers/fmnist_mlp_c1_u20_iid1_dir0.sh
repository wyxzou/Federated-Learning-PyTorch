#!/bin/sh
#SBATCH --account=def-desterck
#SBATCH	--nodes=1
#SBATCH	--gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=11:00:00


REPEATS=1
MODEL=mlp
IID=1
BIDIRECTIONAL=0
EPOCH=100
OPTIMIZER=sparsetopk
DATASET=fmnist
FRAC=1
GPU=cuda:0
USERS=20
LOCAL_BS=10


export SLURM_TMPDIR=/home/wyzou/Federated-Learning-PyTorch/
module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch torchvision tqdm tensorboardX six --no-index

for LR in 0.01 0.008 0.005; do
	for idx in $(seq 1 $REPEATS); do
		srun python src/federated_main.py --model=$MODEL --local_ep=1 --local_bs=$LOCAL_BS --frac=$FRAC --num_users=$USERS --optimizer=$OPTIMIZER \
		 	--bidirectional=$BIDIRECTIONAL --dataset=$DATASET --gpu=$GPU --lr=$LR --iid=$IID --epochs=$EPOCH --number=$idx
	done
done

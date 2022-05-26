#!/bin/sh
#SBATCH --account=def-desterck
#SBATCH	--nodes=1
#SBATCH	--gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=20:00:00


REPEATS=4
MODEL=cnn
IID=0
BIDIRECTIONAL=0
EPOCH=100
OPTIMIZER=sparsetopk
DATASET=mnist
FRAC=1
GPU=cuda:0
USERS=100
LOCAL_BS=10
TOPK=0.001
TOPKD=0.001
MP=1
V=0

export SLURM_TMPDIR=/home/wyzou/Federated-Learning-PyTorch
module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch torchvision tqdm tensorboardX six --no-index

for LR in 0.05 0.07 0.11 0.13; do
	for idx in $(seq 4 $REPEATS); do
		srun python $SLURM_TMPDIR/src/federated_main.py --model=$MODEL --local_ep=1 --local_bs=$LOCAL_BS --frac=$FRAC --num_users=$USERS --optimizer=$OPTIMIZER \
		 	--bidirectional=$BIDIRECTIONAL --dataset=$DATASET --gpu=$GPU --lr=$LR --iid=$IID --epochs=$EPOCH --number=$idx --topk=$TOPK --topk_d=$TOPKD --validation=$V --measure_parameters=$MP
	done
done

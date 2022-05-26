#!/bin/sh

REPEATS=1
MODEL=cnn
IID=1
BIDIRECTIONAL=1
EPOCH=2
OPTIMIZER=sparsetopk
DATASET=fmnist
FRAC=1
GPU=cuda:0
USERS=20
LOCAL_BS=10
LR=0.01
TOPK=0.001
TOPKD=0.001
MP=1

for idx in $(seq 1 $REPEATS); do
	python src/federated_main.py --model=$MODEL --local_ep=1 --local_bs=$LOCAL_BS --frac=$FRAC --num_users=$USERS --optimizer=$OPTIMIZER \
	 	--bidirectional=$BIDIRECTIONAL --dataset=$DATASET --gpu=$GPU --lr=$LR --iid=$IID --epochs=$EPOCH --topk=$TOPK --topk_d=$TOPKD --number=$idx --measure_parameters=$MP;
done

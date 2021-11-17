#!/bin/sh

REPEATS=1
MODEL=vgg
IID=1
BIDIRECTIONAL=1
EPOCH=1
OPTIMIZER=sparsetopk
DATASET=cifar
FRAC=1
GPU=cuda:0
USERS=1
LOCAL_BS=10
LR=0.001

for idx in $(seq 1 $REPEATS); do
	python src/federated_main.py --model=$MODEL --local_ep=1 --local_bs=$LOCAL_BS --frac=$FRAC --num_users=$USERS --optimizer=$OPTIMIZER \
	 	--bidirectional=$BIDIRECTIONAL --dataset=$DATASET --lr=$LR --iid=$IID --epochs=$EPOCH --number=$idx;
done

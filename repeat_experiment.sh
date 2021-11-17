#!/bin/sh

REPEATS=1
MODEL=mlp
IID=0
BIDIRECTIONAL=1
EPOCH=50
OPTIMIZER=sparsetopk
DATASET=mnist
FRAC=0.01
GPU=cuda:0
USERS=100
LOCAL_BS=10

for idx in $(seq 1 $REPEATS); do
	python src/federated_main.py --model=$MODEL --local_ep=1 --local_bs=$LOCAL_BS --frac=$FRAC --num_users=$USERS --optimizer=$OPTIMIZER \
	 	--bidirectional=$BIDIRECTIONAL --dataset=$DATASET --gpu=$GPU --iid=$IID --epochs=$EPOCH --number=$idx;
done

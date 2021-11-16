#!/bin/sh

REPEATS=1
MODEL=mlp
IID=1
BIDIRECTIONAL=1
EPOCH=1
OPTIMIZER=sparsetopk
DATASET=mnist
FRAC=1
GPU=cuda:0
USERS=10

for idx in $(seq 1 $REPEATS); do
	python src/federated_main.py --model=$MODEL --local_ep=1 --frac=$FRAC --num_users=$USERS --optimizer=$OPTIMIZER \
	 	--bidirectional=$BIDIRECTIONAL --dataset=$DATASET --gpu=$GPU --iid=$IID --epochs=$EPOCH --number=$idx;
done

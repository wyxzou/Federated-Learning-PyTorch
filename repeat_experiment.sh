#!/bin/sh

REPEATS=3
MODEL=cnn
IID=1
BIDIRECTIONAL=0
EPOCH=50
OPTIMIZER=sparsetopk
DATASET=cifar
FRAC=1

for idx in $(seq 1 $REPEATS); do
	python src/federated_main.py --model=$MODEL --local_ep=1 --frac=$FRAC --optimizer=$OPTIMIZER \
	 	--bidirectional=$BIDIRECTIONAL --dataset=$DATASET --iid=$IID --epochs=$EPOCH --number=$idx;
done
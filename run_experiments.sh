#!/bin/bash

SEEDS=(1666 1444 1222)
DATASET="cifar10"

for SEED in "${SEEDS[@]}"; do
    echo "=== Running seed $SEED | $DATASET | correlation ==="
    python3 main.py --dataset $DATASET --seed $SEED --use_correlation

    echo "=== Running seed $SEED | $DATASET | baseline ==="
    python3 main.py --dataset $DATASET --seed $SEED
done

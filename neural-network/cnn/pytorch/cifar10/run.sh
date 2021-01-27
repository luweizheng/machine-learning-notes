#!/bin/bash

source activate torch16

python -u train.py  --arch=resnet32 \
                --workers=32 \
                --data="~/Datasets/CIFAR10/" \
                --save-dir="./model" \

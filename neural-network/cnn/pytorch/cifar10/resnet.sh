#!/bin/bash

python -u train.py  --arch=resnet32 \
                --workers=32
                --data="~/Datasets/CIFAR10/cifar-10-batches-py"
                --save-dir=/tmp/resnet-cifar10 \
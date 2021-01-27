#!/bin/bash

python -u train.py  --arch=resnet32 \
                --workers=32
                --save-dir=/tmp/resnet-cifar10 \
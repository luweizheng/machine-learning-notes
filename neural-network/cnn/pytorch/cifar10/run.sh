#!/bin/bash

source activate torch16

currentDir=$(cd "$(dirname "$0")";pwd)
currtime=`date +%Y%m%d%H%M%S`

arch="resnet32"
save_dir=${currentDir}"/"${arch}"/"${currtime}
name="cifar10-"${arch}

echo "save dir:"${save_dir}
echo "running "${name}

python -u train.py  --arch=${arch} \
                --workers=32 \
                --data="~/Datasets/CIFAR10/" \
                --save-dir=${save_dir} \

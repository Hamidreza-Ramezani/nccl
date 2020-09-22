#!/usr/bash

make clean
export DEBUG=1
make -j src.build CUDA_HOME=/mnt/nfs/clustersw/shared/cuda/10.1.243

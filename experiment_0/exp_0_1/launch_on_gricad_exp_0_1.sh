#!/bin/bash

#OAR -n exp_0_1
#OAR -l /nodes=1/gpu=1,walltime=04:00:00
#OAR -p gpumodel='V100'
#OAR --stdout exp_0_1.out
#OAR --stderr exp_0_1.err
#OAR --project pr-gin5_aini

source ../../../environments/ddpm_env/bin/activate

export NUM_GPUS_PER_NODE=1
torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:29501 \
    --nnodes=1 \
    --nproc_per_node=${NUM_GPUS_PER_NODE} \
    /bettik/PROJECTS/pr-gin5_aini/fehrdelt/StrokeUADiag/classification/launch_pipeline.py -c config.json -g ${NUM_GPUS_PER_NODE}
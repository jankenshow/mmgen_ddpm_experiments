#!/bin/bash

# CONFIG=$1
# GPUS=$2
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29500}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     $(dirname "$0")/train.py \
#     $CONFIG \
#     --seed 0 \
#     --launcher pytorch ${@:3}


python train.py \
    ./config/ddpm_cosine_hybird_timestep-4k_drop0.3_customdata_64x64_b8x8_50k.py \
    --seed 0 \
    --launcher none \
    --work-dir ./work_dirs/experiments/ddpm_test_50k/

python train.py \
    config/ddpm_cosine_hybird_timestep-4k_drop0.3_customdata_64x64_b8x4_100k.py \
    --seed 0 \
    --launcher none \
    --work-dir ./work_dirs/experiments/ddpm_test_100k/

python train.py \
    config/ddpm_cosine_hybird_timestep-4k_drop0.3_customdata_128x128_b8x4_150k.py \
    --seed 0 \
    --launcher none \
    --work-dir ./work_dirs/experiments/ddpm_128_test_100k/

python train.py \
    config/ddpm_cosine_hybird_timestep-4k_drop0.3_customdata_256x256_b8x1_100k.py \
    --seed 0 \
    --launcher none \
    --work-dir ./work_dirs/experiments/ddpm_256_test_100k/

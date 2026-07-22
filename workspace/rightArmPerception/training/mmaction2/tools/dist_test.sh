#!/usr/bin/env bash

set -x

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH \
python -m torch.distributed.run \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$PORT" \
    --nproc_per_node="$GPUS" \
    "$(dirname "$0")/test.py" \
    "$CONFIG" \
    "$CHECKPOINT" \
    --launcher pytorch \
    "${@:4}"
# Arguments starting from the forth one are captured by ${@:4}

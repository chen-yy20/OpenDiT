#! /bin/bash
set -x

# export MASTER_ADDR=$(scontrol show jobid=$SLURM_JOB_ID | tr '=' ' ' | grep BatchHost | awk '{print $2}')

export NODE_RANK=$(expr $SLURM_PROCID / $NNODES)
export CUDA_DEVICE_MAX_CONNECTIONS=1 # for async gradient all reduce
# export CUDA_LAUNCH_BLOCKING=1
# export RANK=$SLURM_PROCID
# export LOCAL_RANK=$SLURM_LOCALID

export NUM_GPUS=$GPUS_PER_NODE
export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $(($NUM_GPUS-1)))

exec python -W ignore \
./examples/$SCRIPT \
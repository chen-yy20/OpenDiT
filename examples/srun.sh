#!/bin/bash
set -x

# pjlab cluster config
PARTITION='A800'
export GPUS_PER_NODE='1'
# export NODELIST='g[80]'
export NNODES=1

export MASTER_PORT=$(expr $RANDOM % 10000 + 10000)

# export NNODES=$(scontrol show hostnames ${NODELIST} | wc -l)
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export PYTHONPATH=$PWD:$PYTHONPATH

export MODEL_TYPE="latte"

if [ "$MODEL_TYPE" = "latte" ]; then
        export SCRIPT="latte/sample.py"
        export MODEL_ID="/home/test/test01/cyy/Data/models--maxin-cn--Latte-1/snapshots/0653024365272f061fc44d1078134df22842b687"
elif [ "$MODEL_TYPE" = "cogvideox" ]; then 
        export SCRIPT="cogvideox/sample.py"
        export MODEL_ID="/home/test/test01/cyy/Data/models--THUDM--CogVideoX-2b/snapshots/ad5ce8664edfdc95cdb9773dd4f80048b25f69ac/"
else 
        echo "Invalid MODEL_TYPE: $MODEL_TYPE"
        exit 1
fi

srun \
        --account=test \
        --exclusive=user \
        -p $PARTITION \
        -N $NNODES \
        --time 20:00 \
        --job-name=videosys \
        --ntasks=1 \
        --gres=gpu:$GPUS_PER_NODE \
        --export=ALL \
        bash ./examples/infer.sh


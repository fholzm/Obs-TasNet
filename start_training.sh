#!/bin/bash

# Check if there is one or two arguments
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

# second argument is the training script itself
if [ "$#" -eq 2 ]; then
    TRAINING_SCRIPT=$2
else
    TRAINING_SCRIPT="train_and_evaluate.py"
fi

# Create output filename
FN_OUT=$(basename $1)
FN_OUT="logs/${FN_OUT%.*}.out"

# Start training
nohup python $TRAINING_SCRIPT --config $1 > $FN_OUT 2>&1 &
disown
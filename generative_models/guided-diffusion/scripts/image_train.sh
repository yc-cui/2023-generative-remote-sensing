#!/bin/bash

CONFIG_FILE=$1
CONFIG=$(cat $CONFIG_FILE)

FLAGS=""
while IFS= read -r line; do
    key=$(echo $line | cut -d ':' -f1)
    value=$(echo $line | cut -d ':' -f2- | xargs)
    FLAGS+=" --$key=$value"
done <<< "$CONFIG"

COMMAND="export PYTHONPATH=$PWD; FLAGS=\"$FLAGS\" python scripts/image_train.py \$FLAGS"
echo "Running command: $COMMAND"
eval $COMMAND
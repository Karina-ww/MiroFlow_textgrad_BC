#!/bin/bash
# MiroFlow TextGrad Training Script
# ==================================

# Set environment variables
export OPENAI_API_KEY="sk-zmWyTcLwxNK14gghnlzMG3ih9reb9aBmrdX5DCo76WPEiLBS"
export OPENAI_BASE_URL="https://yibuapi.com/v1"
export LOGGER_LEVEL="INFO"
export CHINESE_CONTEXT="false"

# Training configuration
CONFIG_NAME="train_textgrad_browsecomp_with_memory_claude4"
NUM_EPOCHS=1
MAX_TRAIN_TASKS=200
BATCH_SIZE=8

echo "========================================"
echo "MiroFlow TextGrad Training"
echo "========================================"
echo "Config: $CONFIG_NAME"
echo "Epochs: $NUM_EPOCHS"
echo "Max train tasks per epoch: $MAX_TRAIN_TASKS"
echo "Batch size: $BATCH_SIZE"
echo "========================================"

# Run training
python train_miroflow_textgrad.py \
    --config-name=$CONFIG_NAME \
    train.num_epochs=$NUM_EPOCHS \
    train.max_train_tasks_per_epoch=$MAX_TRAIN_TASKS \
    train.batch_size=$BATCH_SIZE

echo ""
echo "Training complete! Check logs/ for results."

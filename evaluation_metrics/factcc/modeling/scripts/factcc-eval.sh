#! /bin/bash
# Evaluate FactCC model

# UPDATE PATHS BEFORE RUNNING SCRIPT
export CODE_PATH=.\modeling # absolute path to modeling directory
export DATA_PATH= .\data-dev.jsonl  # absolute path to data directory
export CKPT_PATH= .\factcc-checkpoint # absolute path to model checkpoint

export TASK_NAME=factcc_annotated
export MODEL_NAME=bert-base-uncased

python $CODE_PATH/run.py \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_all_checkpoints \
  --do_lower_case \
  --overwrite_cache \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 12 \
  --model_type bert \
  --model_name_or_path $MODEL_NAME \
  --data_dir $DATA_PATH \
  --output_dir $CKPT_PATH

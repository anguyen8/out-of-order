#!/bin/bash
# Functions to run SuperGLUE BERT baselines.
# Usage: ./scripts/superglue-baselines.sh ${TASK} ${GPU_ID} ${SEED}
#   - TASK: one of {"boolq", "commit", "copa", "multirc", "record", "rte", "wic", "wsc"},
#           as well as their *-bow variants and *++ for {"boolq", "commit", "copa", "rte"}
#   - GPU_ID: GPU to use, or -1 for CPU. Defaults to 0.
#   - SEED: random seed. Defaults to 111.

# The base directory for model output.
export JIANT_PROJECT_PREFIX=output
export JIANT_DATA_DIR=superglue_data
export WORD_EMBS_FILE=None
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

function train() {

  TASK_NAME=$1
  GPU=$2
  SEED=$3
  MODEL_BASE=$4
  BATCH_SIZE=$5
  OPTIMIZER=$6
  VAL_INTERVAL=$7
  VAL_DATA_LIMIT=$8

  OUTPUT_DIR=$JIANT_PROJECT_PREFIX/$MODEL_BASE

  PRETRAIN_TASKS=$TASK_NAME
  TARGET_TASKS=$TASK_NAME

  if [[ $TASK_NAME == "rte" ]]; then
    PRETRAIN_TASKS="rte-superglue"
    TARGET_TASKS="rte-superglue"
  elif [[ $TASK_NAME == "wsc" ]]; then
    PRETRAIN_TASKS="winograd-coreference"
    TARGET_TASKS="winograd-coreference"
  fi

  MAX_EPOCHS=10
  LR=.00001

  mkdir -p $OUTPUT_DIR/$TASK_NAME/$SEED/

  ARGUMENTS="random_seed = ${SEED}, "
  ARGUMENTS+="cuda = ${GPU}, "
  ARGUMENTS+="input_module = ${MODEL_BASE}, "
  ARGUMENTS+="exp_name = ${MODEL_BASE}, "
  ARGUMENTS+="run_name = ${TASK_NAME}, "
  ARGUMENTS+="pretrain_tasks = \"${PRETRAIN_TASKS}\", "
  ARGUMENTS+="target_tasks = \"${TARGET_TASKS}\", "
  ARGUMENTS+="do_pretrain = 1, "
  ARGUMENTS+="do_target_task_training = 0, "
  ARGUMENTS+="do_full_eval = 1, "
  ARGUMENTS+="batch_size = ${BATCH_SIZE}, "
  ARGUMENTS+="optimizer = ${OPTIMIZER}, "
  ARGUMENTS+="val_interval = ${VAL_INTERVAL}, "
  ARGUMENTS+="val_data_limit = ${VAL_DATA_LIMIT}, "
  ARGUMENTS+="max_epochs = ${MAX_EPOCHS}, "
  ARGUMENTS+="lr = ${LR}"

  echo $ARGUMENTS
  nohup python -u main.py --config jiant/config/superglue_bert.conf --overrides "${ARGUMENTS}" > $OUTPUT_DIR/$TASK_NAME/$SEED/finetune_logs.txt &
}

function run() {

  TASK_NAME=$1
  GPU=$2
  SEED=$3
  MODEL_BASE=$4
  BATCH_SIZE=$5
  OPTIMIZER=$6
  VAL_INTERVAL=$7
  VAL_DATA_LIMIT=$8
  MODE=$9

  OUTPUT_DIR=$JIANT_PROJECT_PREFIX/$MODEL_BASE

  PRETRAIN_TASKS=$TASK_NAME
  TARGET_TASKS=$TASK_NAME

  if [[ $TASK_NAME == "rte" ]]; then
    PRETRAIN_TASKS="rte-superglue"
    TARGET_TASKS="rte-superglue"
  elif [[ $TASK_NAME == "wsc" ]]; then
    PRETRAIN_TASKS="winograd-coreference"
    TARGET_TASKS="winograd-coreference"
  fi

  # Notes for baseline performance
  # Just need to update run_name to other name rather than model/task name
  # so that the framework will not load the finetuned models. (e.g., add _baseline after task name)
  RUN_NAME=${TASK_NAME}

  if [[ $MODE == "baseline" ]]; then
    RUN_NAME=${TASK_NAME}_baseline
  fi

  MAX_EPOCHS=10
  LR=.00001

  mkdir -p $OUTPUT_DIR/$TASK_NAME/evaluation/$MODE/$SEED/

  ARGUMENTS="random_seed = ${SEED}, "
  ARGUMENTS+="cuda = ${GPU}, "
  ARGUMENTS+="input_module = ${MODEL_BASE}, "
  ARGUMENTS+="exp_name = ${MODEL_BASE}, "
  ARGUMENTS+="run_name = ${TASK_NAME}, "
  ARGUMENTS+="pretrain_tasks = \"${PRETRAIN_TASKS}\", "
  ARGUMENTS+="target_tasks = \"${TARGET_TASKS}\", "
  ARGUMENTS+="do_pretrain = 0, "
  ARGUMENTS+="do_target_task_training = 0, "
  ARGUMENTS+="do_full_eval = 1, "
  ARGUMENTS+="batch_size = ${BATCH_SIZE}, "
  ARGUMENTS+="optimizer = ${OPTIMIZER}, "
  ARGUMENTS+="val_interval = ${VAL_INTERVAL}, "
  ARGUMENTS+="val_data_limit = ${VAL_DATA_LIMIT}, "
  ARGUMENTS+="max_epochs = ${MAX_EPOCHS}, "
  ARGUMENTS+="lr = ${LR}"

  echo $ARGUMENTS
  nohup python -u main.py --config jiant/config/superglue_bert.conf --overrides "${ARGUMENTS}" > $OUTPUT_DIR/$TASK_NAME/evaluation/$MODE/$SEED/predict_logs.txt &

  # THANGPM's NOTES:
  # After the evaluation processes are done, we still need to move predicted files
  # from the model folder output/${MODEL}/${TASK_NAME} to the $OUTPUT folder created above
}

function train_superglue() {
  # MODEL_BASE=bert-base-uncased
  MODEL_BASE=bert-large-cased

  superglue_tasks=(boolq commitbank copa multirc record rte wic wsc)
  batch_sizes=(4 4 4 4 8 4 4 4)
  val_intervals=(1000 60 100 1000 10000 625 1000 139)
  val_data_limits=(5000 5000 5000 -1 -1 5000 5000 5000)
  optimizers=(bert_adam bert_adam bert_adam bert_adam bert_adam bert_adam bert_adam adam)

  for i in "${!superglue_tasks[@]}"; do
    #if [[ $i == 4 ]]; then
    #  continue
    #fi
    train ${superglue_tasks[$i]} $i 42 $MODEL_BASE ${batch_sizes[$i]} ${optimizers[$i]} ${val_intervals[$i]} ${val_data_limits[$i]}
  done
}

train_single_superglue() {
  # MODEL_BASE=bert-base-uncased # bs = 8
  # MODEL_BASE=bert-large-cased # bs = 6
  # MODEL_BASE=roberta-base
  MODEL_BASE=albert-base-v2

  superglue_tasks=(boolq commitbank copa multirc record rte wic wsc)
  batch_sizes=(4 4 4 4 6 4 4 4)
  val_intervals=(1000 60 100 1000 10000 625 1000 139)
  val_data_limits=(5000 5000 5000 -1 -1 5000 5000 5000)
  optimizers=(bert_adam bert_adam bert_adam bert_adam bert_adam bert_adam bert_adam adam)

  i=$1
  train ${superglue_tasks[$i]} $i 42 $MODEL_BASE ${batch_sizes[$i]} ${optimizers[$i]} ${val_intervals[$i]} ${val_data_limits[$i]}
}

function run_superglue() {
  MODE=$1
  MODEL_BASE=bert-base-uncased

  superglue_tasks=(boolq commitbank copa multirc record rte wic wsc)
  batch_sizes=(4 4 4 4 8 4 4 4)
  val_intervals=(1000 60 100 1000 10000 625 1000 139)
  val_data_limits=(5000 5000 5000 -1 -1 5000 5000 5000)
  optimizers=(bert_adam bert_adam bert_adam bert_adam bert_adam bert_adam bert_adam adam)

  for i in "${!superglue_tasks[@]}"; do
    #if [[ $i == 4 ]]; then
    #  continue
    #elif [[ $i == 7 ]]; then
    #  continue
    #fi
    run ${superglue_tasks[$i]} $i 42 $MODEL_BASE ${batch_sizes[$i]} ${optimizers[$i]} ${val_intervals[$i]} ${val_data_limits[$i]} ${MODE}
  done
}

run_single_superglue() {
  i=$1
  MODE=$2

  MODEL_BASE=bert-base-uncased

  superglue_tasks=(boolq commitbank copa multirc record rte wic wsc)
  batch_sizes=(4 4 4 4 6 4 4 4)
  val_intervals=(1000 60 100 1000 10000 625 1000 139)
  val_data_limits=(5000 5000 5000 -1 -1 5000 5000 5000)
  optimizers=(bert_adam bert_adam bert_adam bert_adam bert_adam bert_adam bert_adam adam)

  run ${superglue_tasks[$i]} $i 42 $MODEL_BASE ${batch_sizes[$i]} ${optimizers[$i]} ${val_intervals[$i]} ${val_data_limits[$i]} ${MODE}
}

#train_superglue
train_single_superglue 3

# MODEL in [baseline, original, shuffled]
#run_superglue original
#run_single_superglue 7 original

# NOTES FOR RUNNING COMMAND LINE
# (jiant) thang@gpu2:/mnt/raid/thang/Projects/open_sources/jiant$ bash scripts/run_superglue.sh
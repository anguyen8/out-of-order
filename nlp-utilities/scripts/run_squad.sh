
function train_squad() {

  VERSION=$1
  MODEL=$2
  OUTPUT_DIR=$3
  TRAIN_BATCH_SIZE=$4
  EVAL_BATCH_SIZE=$5
  SEED=$6
  GPU=$7

  OUTPUT_DIR=${OUTPUT_DIR}/${MODEL}-finetuned
  mkdir -p $OUTPUT_DIR
  export CUDA_VISIBLE_DEVICES=$GPU

  params=()
  params+=(--do_train)
  params+=(--num_train_epochs 2.0)
  params+=(--save_steps 5000)

  SQUAD_DIR=../squad/v${VERSION}
  TRAIN_FILE_NAME="train-v1.1.json"
  DEV_FILE_NAME="dev-v1.1.json"

  if [[ ${VERSION} == 2 ]]; then
    TRAIN_FILE_NAME="train-v2.0.json"
    DEV_FILE_NAME="dev-v2.0.json"

    params+=(--version_2_with_negative)
  fi

  params+=(--model_type "bert")
  params+=(--model_name_or_path "${MODEL}")
  params+=(--do_eval)
  params+=(--do_lower_case)
  params+=(--train_file "${SQUAD_DIR}/${TRAIN_FILE_NAME}")
  params+=(--predict_file "${SQUAD_DIR}/${DEV_FILE_NAME}")
  params+=(--per_gpu_train_batch_size "${TRAIN_BATCH_SIZE}")
  params+=(--per_gpu_eval_batch_size "${EVAL_BATCH_SIZE}")
  params+=(--learning_rate 3e-5)
  params+=(--max_seq_length 384)
  params+=(--doc_stride 128)
  params+=(--output_dir "${OUTPUT_DIR}")
  params+=(--overwrite_output_dir)
  params+=(--overwrite_cache)
  params+=(--seed "${SEED}")

  nohup python -u run_squad.py "${params[@]}" > $OUTPUT_DIR/finetune_logs.txt &
}

function run_squad() {

  VERSION=$1
  MODEL=$2
  OUTPUT_DIR=$3
  TRAIN_BATCH_SIZE=$4
  EVAL_BATCH_SIZE=$5
  SEED=$6
  GPU=$7
  MODE=$8

  OUTPUT_DIR=${OUTPUT_DIR}/${MODEL_BASE}-evaluation/${MODE}
  mkdir -p $OUTPUT_DIR
  export CUDA_VISIBLE_DEVICES=$GPU

  params=()

  SQUAD_DIR=../squad/v${VERSION}
  TRAIN_FILE_NAME="train-v1.1.json"
  DEV_FILE_NAME="dev-v1.1.json"

  if [[ ${VERSION} == 2 ]]; then
    TRAIN_FILE_NAME="train-v2.0.json"
    DEV_FILE_NAME="dev-v2.0.json"

    params+=(--version_2_with_negative)
    params+=(--save_steps 5000)
  fi

  if [[ ${MODE} == "shuffled" ]]; then
    params+=(--shuffle_data)
  fi

  params+=(--model_type "bert")
  params+=(--model_name_or_path "${MODEL}")
  params+=(--do_eval)
  params+=(--do_lower_case)
  params+=(--train_file "${SQUAD_DIR}/${TRAIN_FILE_NAME}")
  params+=(--predict_file "${SQUAD_DIR}/${DEV_FILE_NAME}")
  params+=(--per_gpu_train_batch_size "${TRAIN_BATCH_SIZE}")
  params+=(--per_gpu_eval_batch_size "${EVAL_BATCH_SIZE}")
  params+=(--learning_rate 3e-5)
  params+=(--max_seq_length 384)
  params+=(--doc_stride 128)
  params+=(--output_dir "${OUTPUT_DIR}")
  params+=(--overwrite_output_dir)
  params+=(--overwrite_cache)
  params+=(--seed "${SEED}")

  nohup python -u run_squad.py "${params[@]}" > $OUTPUT_DIR/predict_logs.txt &
}

function do_finetuning() {
  echo "*** FINETUNING STARTED ***"
  train_squad 1 $MODEL_BASE ../results/squad/v1 12 12 42 0
  train_squad 2 $MODEL_BASE ../results/squad/v2 12 12 42 1
}

function do_evaluation() {
  echo "*** EVALUATION STARTED ***"

  # For baseline
  run_squad 1 $MODEL_BASE ../results/squad/v1 12 12 42 0 baseline
  # For original + shuffled: Change shuffle mode in line 716
  run_squad 1 ../results/squad/v1/$MODEL_BASE-finetuned ../results/squad/v1 12 12 42 1 original
  run_squad 1 ../results/squad/v1/$MODEL_BASE-finetuned ../results/squad/v1 12 12 42 2 shuffled

  # For baseline
  run_squad 2 $MODEL_BASE ../results/squad/v2 12 12 42 3 baseline
  # For original + shuffled: Change shuffle mode in line 716
  run_squad 2 twmkn9/bert-base-uncased-squad2 ../results/squad/v2 12 12 42 4 original
  run_squad 2 twmkn9/bert-base-uncased-squad2 ../results/squad/v2 12 12 42 5 shuffled
}

# export MODEL_BASE=bert-base-uncased
export MODEL_BASE=roberta-base
#export MODEL_BASE=albert-base-v2

do_finetuning
#do_evaluation

# V1
# bert-large-uncased-whole-word-masking:
# bert-large-uncased-whole-word-masking-finetuned-squad: Original: 93.16 / 86.93 || Shuffled: 74.88 / 65.84

# bert-base-uncased: 6.86 / 0.057
# bert-base-uncased-finetuned: Original: / || Shuffled: /

# V2
# bert-base-uncased: 50.07 / 50.07
# twmkn9/bert-base-uncased-squad2:  Original: 75.83 / 72.37  || Shuffled: 64.43 / 60.89
# twmkn9/albert-base-v2-squad2:     Original: 81.45 / 77.94

#run_squad_v2 twmkn9/bert-base-uncased-squad2 ../results/squad/v2 8 8 42
#run_squad_v2 bert-base-uncased ../results/squad/v2 8 8 42


# NOTE FOR RUNNING WITH COMMAND LINE
# (finetuning) thang@gpu2:/mnt/raid/thang/Projects/finetuning/src$ bash run_squad.sh
# MODE should be in [baseline, original, shuffled]

export GLUE_DIR=/home/home2/thang/Projects/open_sources/transformer_gp2/examples/glue_data


function run() {

  local TASK_NAME=$1
  local BASE_DIR=$2
  local GPU=$3
  local SEED=$4
  local MODEL_BASE=$5
  local ANALYZER=$6
  local CHECKED_POINT=$7
  local TOP_N=$8
  local MAX_SEQ_LEN=$9

  # ============================================================================================================
  local OUTPUT_DIR=pickle_files/${MODEL_BASE}/${TASK_NAME}/${ANALYZER}/MLM-${MASKED_LANGUAGE_MODEL}
  if [[ ${ANALYZER} != "RIS" ]]; then
    local OUTPUT_DIR=pickle_files/${MODEL_BASE}/${TASK_NAME}/${CHECKED_POINT}/${ANALYZER} #/${SEED} # SEED is only for sanity check
  fi
  echo ${OUTPUT_DIR}
  # ============================================================================================================

  mkdir -p $OUTPUT_DIR
  export CUDA_VISIBLE_DEVICES=$GPU

  local params=()

  local EVAL_BATCH_SIZE=256
  if [[ ${MAX_SEQ_LEN} == 512 ]]; then
    local EVAL_BATCH_SIZE=16
  fi

  # ThangPM: Automatically load pretrained model
  # ============================================================================================================
#  local MODEL_NAME_OR_PATH=${BASE_DIR}/${MODEL_BASE}/${TASK_NAME}/finetuned             # Normal finetuning
#  local MODEL_NAME_OR_PATH=pmthangk09/bert-base-uncased-esnli
#  local MODEL_NAME_OR_PATH=pmthangk09/bert-base-uncased-superglue-multirc
#  local MODEL_NAME_OR_PATH=pmthangk09/bert-base-uncased-glue-sst2

  local MODEL_NAME_OR_PATH=${BASE_DIR}/bert-base-uncased/${TASK_NAME}/finetuned               # For SST-2 and MultiRC
#  local MODEL_NAME_OR_PATH=${BASE_DIR}/${MODEL_BASE}/${TASK_NAME}/finetuned_${MAX_SEQ_LEN}    # For ESNLI

  # Loading checkpoints if not final version
  if [[ ${CHECKED_POINT} != *"final"* ]]; then
    local MODEL_NAME_OR_PATH=${BASE_DIR}/${MODEL_BASE}/${TASK_NAME}/finetuned/${CHECKED_POINT}
  fi

  if [[ ${MODEL_BASE} == *"extra"* ]]; then
    local MODEL_NAME_OR_PATH=${BASE_DIR}/${MODEL_BASE}/${SEED}/${TASK_NAME}/finetuned   # Extra-finetuning
  fi
  # ============================================================================================================

  params+=(--model_name_or_path "${MODEL_NAME_OR_PATH}")
  params+=(--task_name "${TASK_NAME}")
  params+=(--do_eval)
  params+=(--data_dir "${GLUE_DIR}/${TASK_NAME}")
  params+=(--max_seq_length "${MAX_SEQ_LEN}")
  params+=(--per_device_eval_batch_size "${EVAL_BATCH_SIZE}")
  params+=(--output_dir "${OUTPUT_DIR}")
  params+=(--overwrite_output_dir)
  params+=(--seed "${SEED}")
  params+=(--model_base "${MODEL_BASE}")
  params+=(--masked_lm "${MASKED_LANGUAGE_MODEL}")
  params+=(--analyzer "${ANALYZER}")
  params+=(--checkpoint "${CHECKED_POINT}")

  # Otherwise, use this:
#  if [[ ${ANALYZER} != "RIS" ]]; then
#    nohup python -u run_analyzers_lite.py "${params[@]}" > $OUTPUT_DIR/run_analyzers_logs.txt &
#  else
#    nohup python -u run_analyzers_lite.py "${params[@]}" > $OUTPUT_DIR/run_analyzers_logs_${TOP_N}.txt &
#  fi

  echo "${params[@]}"

  # run_analyzers_logs
  # run_analyzers_logs_cont
  # run_evaluation_logs
  # run_evaluation_roar_logs
#  nohup python -u run_analyzers_lite.py "${params[@]}" > $OUTPUT_DIR/run_evaluation_roar_baseline_logs.txt &
#  nohup python -u run_analyzers_lite.py "${params[@]}" > $OUTPUT_DIR/run_evaluation_roar_logs_2nd.txt &
  nohup python -u run_analyzers_lite.py "${params[@]}" > $OUTPUT_DIR/run_evaluation_roar_bert_logs.txt &  # _${TOP_N}
}

function run_analyzers() {

  echo "*** ANALYZERS STARTED ***"

  local i=$1
  local SEED=$2
  local GPU=$3
  local ANALYZER=$4
  local CHECKED_POINT=$5
  local TOP_N=$6
  local MAX_SEQ_LEN=$7

  if [[ ${GPU} -gt 7 ]]; then
    local GPU=0
  fi

  local BASE_DIR=../../examples/models
  local glue_tasks=(CoLA SST-2 MRPC STS-B QQP MNLI QNLI RTE WNLI SST ESNLI MultiRC)

  run ${glue_tasks[$i]} ${BASE_DIR} ${GPU} ${SEED} ${MODEL_BASE} ${ANALYZER} ${CHECKED_POINT} ${TOP_N} ${MAX_SEQ_LEN}
}

function run_all_RIS_versions() {
  local TASK_IDX=$1
  local ANALYZER=$2
  local TOP_N=$3

  local MODEL_BASES=(roberta-base roberta-extra-finetune)
  local MASKED_LMS=(bert-base-uncased) # roberta-base

  local GPUS=(1 2)
#  local GPUS=(0 1 2 3)
#  local GPUS=(4 5 6 7)

  for i in "${!MODEL_BASES[@]}"; do
    for j in "${!MASKED_LMS[@]}"; do

      local SEED=42
      if [[ ${MODEL_BASES[$i]} == *"extra"* ]]; then
        local SEED=200
      fi

      export MASKED_LANGUAGE_MODEL=${MASKED_LMS[$j]}
      export MODEL_BASE=${MODEL_BASES[$i]}

#      run_analyzers ${TASK_IDX} ${SEED} ${GPUS[$i]} ${ANALYZER} ${TOP_N}
      echo ${TASK_IDX} ${SEED} ${GPUS[$i]} ${ANALYZER} $i $j
    done
  done
}

function run_all_baseline_versions() {
  local TASK_IDX=$1
  local ANALYZER=$2
  local TOP_N=$3

  local MODEL_BASES=(roberta-base roberta-extra-finetune)
#  local GPUS=(0 1)
  local GPUS=(2 3)

  for i in "${!MODEL_BASES[@]}"; do

    local SEED=42
    if [[ ${MODEL_BASES[$i]} == *"extra"* ]]; then
      local SEED=200
    fi

    export MASKED_LANGUAGE_MODEL=bert-base-uncased
    export MODEL_BASE=${MODEL_BASES[$i]}

    run_analyzers ${TASK_IDX} ${SEED} ${GPUS[$i]} ${ANALYZER} ${TOP_N}
#      echo ${TASK_IDX} ${SEED} ${GPUS[$i]} ${ANALYZER}
  done
}

function run_one_analyzer() {
  local TASK_IDX=$1
  export MASKED_LANGUAGE_MODEL=bert-base-uncased  # vocab_size ~ 30K
#  export MASKED_LANGUAGE_MODEL=roberta-base       # vocab_size ~ 50k

  # BASE
#  export MODEL_BASE=roberta-base
  export MODEL_BASE=bert-base-uncased

#  run_analyzers ${TASK_IDX} 42 0 RIS              # 0 mlm-bert and 1 mlm-roberta
  #run_analyzers ${TASK_IDX} 42 2 OccEmpty
  run_analyzers ${TASK_IDX} 42 1 OccZero
#  run_analyzers ${TASK_IDX} 42 0 InputMargin

  # EXTRA FINETUNING
  #export MODEL_BASE=roberta-extra-finetune        # 2 mlm-bert and 3 mlm-roberta
  #run_analyzers ${TASK_IDX} 200 2 RIS
  #run_analyzers ${TASK_IDX} 200 3 OccEmpty
  #run_analyzers ${TASK_IDX} 200 ? OccZero
}

function run_ALL_analyzers() {
  local TASK_IDX=$1

#  export MASKED_LANGUAGE_MODEL=../../examples/models/bert-base-uncased-sst2  # vocab_size ~ 30K
#  export MODEL_BASE=bert-base-uncased-sst2

  export MASKED_LANGUAGE_MODEL=bert-base-uncased  # vocab_size ~ 30K
  export MODEL_BASE=bert-base-uncased
#  export MODEL_BASE=bert-base-uncased-conditionally-1000  # bert-base-uncased OR roberta-base OR roberta-extra-finetune

  local ANALYZERS=(LIME LIME-BERT LIME-BERT-SST2)    # InputMargin OccZero OccUnk OccEmpty LIME LIME-BERT LIME-BERT-SST2
  local CHECKED_POINTS=(final_dev final_train)   # checkpoint-300 checkpoint-600 final final_dev final_train
  local GPUS=(0 1 2 3 4 5)
  local SEED=42                          # 79: seed for ESNLI only
  local TOP_N=10
  local MAX_SEQ_LEN=128

  for i in "${!ANALYZERS[@]}"; do
    for j in "${!CHECKED_POINTS[@]}"; do
      run_analyzers ${TASK_IDX} ${SEED} ${GPUS[$i*2 + $j]} ${ANALYZERS[$i]} ${CHECKED_POINTS[$j]} ${TOP_N} ${MAX_SEQ_LEN}
#      echo ${TASK_IDX} ${SEED} ${GPUS[$i*2 + $j]} ${ANALYZERS[$i]} ${CHECKED_POINTS[$j]} # ${GPUS[$i]} OR ${GPUS[$i*2 + $j]}
    done
  done
}

#run_one_analyzer 7
#run_one_analyzer 1

# SST-2: 1, QQP: 4, RTE: 7
#run_all_baseline_versions 1 OccEmpty deletion
#run_all_baseline_versions 7 OccEmpty deletion

#run_all_baseline_versions 1 OccZero deletion
#run_all_baseline_versions 7 OccZero deletion

#run_all_RIS_versions 1 InputMargin threshold
#run_all_RIS_versions 7 InputMargin deletion

run_ALL_analyzers 1 # 1 9 10 11
#run_all_RIS_versions 1 InputMargin threshold
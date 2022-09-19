export GLUE_DIR=/mnt/raid/thang/Projects/open_sources/transformers/examples/glue_data
export TASK_NAME=CoLA

python text-classification/run_glue.py \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --logging_steps 500 \
  --logging_dir runs/DataAug/$TASK_NAME/DN/ \
  --evaluate_during_training \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir models/DataAug/$TASK_NAME/DN/ \
  --overwrite_output_dir
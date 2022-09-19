# training on 4 tesla V100(16GB) GPUS

export SWAG_DIR=/mnt/raid/thang/Projects/open_sources/transformers/examples/SWAG_thangpm/data

python run_multiple_choice.py \
--task_name swag \
--model_name_or_path bert-base-uncased \
--do_train \
--do_eval \
--data_dir $SWAG_DIR \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--max_seq_length 80 \
--output_dir models/swag_customized_bert_base \
--per_gpu_eval_batch_size=16 \
--per_device_train_batch_size=16 \
--gradient_accumulation_steps 2 \
--overwrite_output

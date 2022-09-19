export SQUAD_DIR=/mnt/raid/thang/Projects/open_sources/transformers/examples/thangpm/SQuAD

#python run_squad.py \
#  --model_type bert \
#  --model_name_or_path models/bert_large_uncased_fine_tuned_10_epochs/checkpoint-220000 \
#  --do_eval \
#  --train_file $SQUAD_DIR/train-v1.1.json \
#  --predict_file $SQUAD_DIR/dev-v1.1.json \
#  --per_gpu_train_batch_size 3 \
#  --per_gpu_eval_batch_size 3 \
#  --learning_rate 3e-5 \
#  --num_train_epochs 2.0 \
#  --max_seq_length 384 \
#  --doc_stride 128 \
#  --output_dir ../models/bert_large_uncased_finetuned_10_epochs_evaluation/

#python run_squad.py \
#  --model_type bert \
#  --model_name_or_path "bert-large-uncased-whole-word-masking-finetuned-squad" \
#  --do_eval \
#  --train_file $SQUAD_DIR/train-v1.1.json \
#  --predict_file $SQUAD_DIR/dev-v1.1.json \
#  --per_gpu_train_batch_size 3 \
#  --per_gpu_eval_batch_size 3 \
#  --learning_rate 3e-5 \
#  --num_train_epochs 2.0 \
#  --max_seq_length 384 \
#  --doc_stride 128 \
#  --output_dir ../models/wwm_uncased_finetuned_squad_v1_evaluation_1st/

#python -m torch.distributed.launch --nproc_per_node=8 run_squad.py \
#    --model_type bert \
#    --model_name_or_path "bert-large-uncased-whole-word-masking-finetuned-squad" \
#    --do_eval \
#    --train_file $SQUAD_DIR/train-v1.1.json \
#    --predict_file $SQUAD_DIR/dev-v1.1.json \
#    --learning_rate 3e-5 \
#    --num_train_epochs 2 \
#    --max_seq_length 384 \
#    --doc_stride 128 \
#    --output_dir ../models/wwm_uncased_finetuned_squad_v1_evaluation/ \
#    --per_gpu_eval_batch_size=3   \
#    --per_gpu_train_batch_size=3   \


# python -m torch.distributed.launch --nproc_per_node=8 run_squad.py \
#python run_squad.py \
#     --model_type bert \
#     --model_name_or_path bert-large-uncased-whole-word-masking \
#     --do_train \
#     --do_eval \
#     --do_lower_case \
#     --train_file $SQUAD_DIR/train-v1.1.json \
#     --predict_file $SQUAD_DIR/dev-v1.1.json \
#     --learning_rate 3e-5 \
#     --num_train_epochs 10 \
#     --max_seq_length 384 \
#     --doc_stride 128 \
#     --output_dir ../models/wwm_uncased_finetuned_10_epochs/ \
#     --per_gpu_eval_batch_size 3   \
#     --per_gpu_train_batch_size 3

#python run_squad.py \
#  --model_type bert \
#  --model_name_or_path bert-large-uncased \
#  --do_train \
#  --do_eval \
#  --do_lower_case \
#  --train_file $SQUAD_DIR/train-v1.1.json \
#  --predict_file $SQUAD_DIR/dev-v1.1.json \
#  --per_gpu_train_batch_size 3 \
#  --per_gpu_eval_batch_size 3 \
#  --learning_rate 3e-5 \
#  --num_train_epochs 10.0 \
#  --max_seq_length 384 \
#  --doc_stride 128 \
#  --output_dir ../models/bert_large_uncased_fine_tuned_10_epochs/

# *************** SQUAD v2 ***************

# twmkn9/bert-base-uncased-squad2: 50.09 / 50.08
# a-ware/roberta-large-squadv2:
# twmkn9/albert-base-v2-squad2
# mrm8488/roberta-base-1B-1-finetuned-squadv2:
# mrm8488/t5-base-finetuned-squadv2: NEED TO UPDATE TRANSFORMER for T5 (81.32 / 77.64)

python run_squad.py \
  --model_type bert \
  --model_name_or_path "twmkn9/albert-base-v2-squad2" \
  --do_eval \
  --train_file $SQUAD_DIR/train-v2.0.json \
  --predict_file $SQUAD_DIR/dev-v2.0.json \
  --per_gpu_train_batch_size 3 \
  --per_gpu_eval_batch_size 3 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ../models/albert_base_squad_v2_evaluation/

#python run_squad.py \
#  --model_type bert \
#  --model_name_or_path models/wwm_uncased_finetuned_squad_v2 \
#  --do_eval \
#  --train_file $SQUAD_DIR/train-v2.0.json \
#  --predict_file $SQUAD_DIR/dev-v2.0.json \
#  --per_gpu_train_batch_size 6 \
#  --per_gpu_eval_batch_size 3 \
#  --learning_rate 3e-5 \
#  --num_train_epochs 2.0 \
#  --max_seq_length 384 \
#  --doc_stride 128 \
#  --output_dir ../models/wwm_uncased_finetuned_squad_v2_evaluation/

# python -m torch.distributed.launch --nproc_per_node=8 run_squad.py \
#     --model_type bert \
#     --model_name_or_path bert-large-uncased-whole-word-masking \
#     --do_train \
#     --do_eval \
#     --train_file $SQUAD_DIR/train-v2.0.json \
#     --predict_file $SQUAD_DIR/dev-v2.0.json \
#     --learning_rate 3e-5 \
#     --num_train_epochs 2 \
#     --max_seq_length 384 \
#     --doc_stride 128 \
#     --output_dir ../models/wwm_uncased_finetuned_squad_v2_2/ \
#     --per_gpu_eval_batch_size=2   \
#     --per_gpu_train_batch_size=2   \
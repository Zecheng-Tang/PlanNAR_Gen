export CUDA_VISIBLE_DEVICES=0
export HF_DATASETS_CACHE="/data1/tzc/huggingface"
TRAIN_FILE=$1
VALID_FILE=$2
SAVE_PATH=$3


# python -m torch.distributed.launch --nproc_per_node 8 --use_env run_mlm_no_trainier.py \
python process.py \
--output_dir "/data/tzc/model/T5/test" \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 4 \
--learning_rate 5e-5 \
--model_name_or_path "t5-base" \
--config_name "t5-base" \
--tokenizer_name "t5-base" \
--train_file ${TRAIN_FILE} \
--validation_file ${VALID_FILE} \
--save_path ${SAVE_PATH} \
--max_seq_length 512 \
--lr_scheduler_type "cosine" \
--num_warmup_steps 4000 \
--weight_decay 0.001 \
--preprocessing_num_workers 100 \
--mlm_probability 0.15 \
--load_from_cache False \
--logging_steps 2000 \
--mean_noise_span_length 3.0 \
--num_train_epochs 3

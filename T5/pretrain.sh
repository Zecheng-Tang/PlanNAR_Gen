export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_DATASETS_CACHE="/data1/tzc/huggingface"
TRAIN_FILE=/data1/tzc/data/pretrain/c4/enPro/concact/bin/500_sep/train
VALID_FILE=/data1/tzc/data/pretrain/c4/enPro/concact/bin/500_sep/valid
# python run_mlm_no_trainier.py \

python -m torch.distributed.launch --nproc_per_node 8 --use_env run_mlm_no_trainier.py \
        --output_dir "/data1/tzc/model/pretrain/T5-base/500_sep" \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --learning_rate 5e-5 \
        --model_name_or_path "/data1/tzc/model/pretrain/T5-base/500_sep/24000" \
        --config_name "/data1/tzc/model/pretrain/T5-base/500_sep/24000" \
        --tokenizer_name "/data1/tzc/model/pretrain/T5-base/500_sep/24000" \
        --train_save ${TRAIN_FILE} \
        --validation_save ${VALID_FILE} \
        --max_seq_length 512 \
        --lr_scheduler_type "cosine" \
        --num_warmup_steps 4000 \
        --weight_decay 0.001 \
        --preprocessing_num_workers 100 \
        --mlm_probability 0.15 \
        --load_from_cache False \
        --logging_steps 4000 \
        --mean_noise_span_length 3.0 \
        --num_train_epochs 3 \
        --save_steps 8000 \

CUDA_VISIBLE_DEVICES=0

INPUT=$1
OUTPUT=$2

python generation.py \
    --tokenizer_name_or_path /data1/tzc/model/T5/story \
    --model_name_or_path /data1/tzc/model/T5/story \
    --prompt_file ${INPUT} \
    --output_file ${OUTPUT}

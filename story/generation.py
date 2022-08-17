import argparse
import logging
import time
import numpy as np
import torch
from transformers import(
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    AutoConfig,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def prepare_input(args, prompt_text):
    prefix = args.prefix if args.prefix is not None else ""
    prompt_text = prefix + prompt_text
    return prompt_text


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained tokenizer or shortcut name",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument("--prompt_file", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")
    parser.add_argument("--output_file", type=str, default=None, help="")

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--p", type=float, default=1.0)

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    set_seed(args)
        
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    if len(tokenizer.additional_special_tokens) == 0:
        # special_tokens_dict = {
        #     'additional_special_tokens': ["[SOT]", "[EOT]", "[SEP]"]}
        # tokenizer.add_special_tokens(special_tokens_dict)
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.resize_token_embeddings(len(tokenizer))
    if args.fp16:
        model.half()

    # args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_length)
    logger.info(args)
    generated_sequences = []
    with open(args.prompt_file,"r") as f:
        prompts=f.readlines()
        f.close()
    fout=open(args.output_file,"w")
    used_time=[]
    for prompt_text in prompts:
        prompt_text=prompt_text.strip()
        encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(args.device)
        input_ids = encoded_prompt
        start_time=time.time()
        while True:
            output_sequences = model.generate(input_ids=input_ids)
            # print(output_sequences)

            generated_sequence=output_sequences[0]
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(tokenizer.pad_token) if args.stop_token else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
            text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            )
            # total_sequence=total_sequence.replace("\n","").split(".")
            # if len(total_sequence)>=5:
            #     break
            break
        # total_sequence=".".join(total_sequence[:4])+"."
        end_time=time.time()
        used_time.append(end_time-start_time)
        print(end_time-start_time)
        generated_sequences.append([prompt_text,total_sequence])
        fout.write("ipt:"+prompt_text.strip()+"\n")
        fout.write("opt:"+total_sequence.replace("<|endoftext|>","").strip()+"\n")
        fout.flush()
    print("average_time:",sum(used_time)/len(used_time))
    fout.close()
    return generated_sequences


if __name__ == "__main__":
    main()
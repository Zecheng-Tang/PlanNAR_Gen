'''
created by tzc
2022/4/1
'''

import os
from posixpath import split
from datasets import load_dataset, load_metric
import torch
import argparse
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(description="For model inference")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--inference_file", type=str, default=None, help="A csv or a json file containing the inference data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the inference results."
    )
    parser.add_argument(
        "--metric", 
        type=str, 
        default='ROUGE', 
        help="Define the evaluation metric"
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


    batch = [tokenizer(s, return_tensors='pt', padding='max_length', max_length=inp_max_len) for s in sents]


if __name__ == '__main__':
    args = parse_args()

    # initilize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(args.tokenizer_name)
    model.eval()
    model.cuda()

    # load datasets
    if args.dataset_name is not None:
        # Downloading and loading a inference(test) dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, split='test')
    else:
        data_files = {}
        if args.inference_file is not None:
            data_files["test"] = args.inference_file
        extension = args.inference_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    # tokenize datasets
    


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path')
parser.add_argument('-i', '--input_path')
parser.add_argument('-o', '--output_path')
parser.add_argument('--bea19', action='store_true')
args = parser.parse_args()


tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained(args.model_path, force_bos_token_to_be_generated=True)
model.eval()
model.cuda()


def run_model(sents):
    num_ret_seqs = 10
    inp_max_len = 33
    batch = [tokenizer(s, return_tensors='pt', padding='max_length', max_length=inp_max_len) for s in sents]
    oidx2bidx = {} #original index to final batch index
    final_batch = []
    for oidx, elm in enumerate(batch):
        if elm['input_ids'].size(1) <= inp_max_len:
            oidx2bidx[oidx] = len(final_batch)
            final_batch.append(elm)
    batch = {key: torch.cat([elm[key] for elm in final_batch], dim=0) for key in final_batch[0]}
    with torch.no_grad():
        generated_ids = model.generate(batch['input_ids'].cuda(),
                                attention_mask=batch['attention_mask'].cuda(),
                                num_beams=10, num_return_sequences=num_ret_seqs, max_length=65)
    _out = tokenizer.batch_decode(generated_ids.detach().cpu(), skip_special_tokens=True)
    outs = []
    for i in range(0, len(_out), num_ret_seqs):
        outs.append(_out[i:i+num_ret_seqs])
    final_outs = [[sents[oidx]] if oidx not in oidx2bidx else outs[oidx2bidx[oidx]] for oidx in range(len(sents))]
    return final_outs


def run_for_wiki_yahoo_conll():
    sents = [detokenize_sent(l.strip()) for l in open(args.input_path)]
    b_size = 200
    outs = []
    for j in tqdm(range(0, len(sents), b_size)):
        sents_batch = sents[j:j+b_size]
        outs_batch = run_model(sents_batch)
        for sent, preds in zip(sents_batch, outs_batch):
            preds_detoked = [detokenize_sent(pred) for pred in preds]
            preds = [' '.join(spacy_tokenize_gec(pred)) for pred in preds_detoked]
            outs.append({'src': sent, 'preds': preds})
    os.system('mkdir -p {}'.format(os.path.dirname(args.output_path)))
    with open(args.output_path, 'w') as outf:
        for out in outs:
            print (out['preds'][0], file=outf)


def run_for_bea19():
    sents = [detokenize_sent(l.strip()) for l in open(args.input_path)]
    b_size = 200
    outs = []
    for j in tqdm(range(0, len(sents), b_size)):
        sents_batch = sents[j:j+b_size]
        outs_batch = run_model(sents_batch)
        for sent, preds in zip(sents_batch, outs_batch):
            preds_detoked = [detokenize_sent(pred) for pred in preds]
            preds = [' '.join(spacy_tokenize_bea19(pred)) for pred in preds_detoked]
            outs.append({'src': sent, 'preds': preds})
    os.system('mkdir -p {}'.format(os.path.dirname(args.output_path)))
    with open(args.output_path, 'w') as outf:
        for out in outs:
            print (out['preds'][0], file=outf)


if args.bea19:
    run_for_bea19()
else:
    run_for_wiki_yahoo_conll()

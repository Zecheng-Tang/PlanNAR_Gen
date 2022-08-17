import sys
import nltk
import numpy as np
from nltk import ngrams
from nltk.tokenize import WordPunctTokenizer
import os
import copy
def bleu(refs, cands):
    result = {}
    for i in range(1, 5):
        result["bleu-%d"%i] = "%.4f"%(nltk.translate.bleu_score.corpus_bleu([[r] for r in refs], cands, weights=tuple([1./i for j in range(i)])))
    return result
def distinct(cands):
    result = {}
    for i in range(1, 5):
        num, all_ngram, all_ngram_num = 0, {}, 0.
        for k, cand in enumerate(cands):
            ngs = ["_".join(c) for c in ngrams(cand, i)]
            all_ngram_num += len(ngs)
            for s in ngs:
                if s in all_ngram:
                    all_ngram[s] += 1
                else:
                    all_ngram[s] = 1
        result["distinct-%d"%i] = "%.4f"%(len(all_ngram) / float(all_ngram_num))
    return result
ipt=[]
cand=[]
truth=[]
title=[]
with open("/data0/data/fkq/dataset/roc/test.source","r") as fin:
    ipt=[line.strip() for line in fin]
with open("/data0/data/fkq/dataset/roc/test.target","r") as fin:
    truth=[line.strip() for line in fin]
with open("/data0/data/fkq/result/gpt2_early_exit.txt","r") as fin:
    for i, line in enumerate(fin):
        if (i+1)%2==0:
            cand.append(line.split("opt:")[1].strip().replace("<|endoftext|>",""))
        elif (i+1)%2==1:
            title.append(line.split("ipt:")[1].strip())
        # cand.append(line.split("opt:")[1].strip().lower())
print(len(cand))
ipt=ipt[:len(cand)]
truth=truth[:len(cand)]
for i in range(10):
    print("ipt:{}     titile:{}" .format(ipt[i],title[i]))
tokenizer=WordPunctTokenizer()
ipt_token, truth_token, cand_token= [tokenizer.tokenize(i) for i in ipt], [tokenizer.tokenize(t) for t in truth], [tokenizer.tokenize(c) for c in cand]
print(bleu(truth_token,cand_token))
print(distinct(cand_token))
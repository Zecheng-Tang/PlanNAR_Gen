from lib2to3.pgen2 import token
import logging
import math
import os
import sys
from turtle import forward
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import copy
from transformers import (
    AutoTokenizer, 
    AutoModel,
    PreTrainedModel,
    RobertaForCausalLM
)
from discriminator import Max_Discriminator
import numpy as np


class PlanNAT(PreTrainedModel):
    def __init__(self, dec_config, tokenizer, args):
        super(PlanNAT, self).__init__(dec_config, tokenizer, args)

        self.vocab_size = tokenizer.vocab_size
        # for encoder, we recommand sentence-bert
        self.encoder = AutoModel.from_pretrained(args.enc_name_or_path)._resize_token_embeddings(len(tokenizer))
        self.decoder = RobertaForCausalLM.from_pretrained(args.dec_name_or_path, dec_config)._resize_token_embeddings(len(tokenizer))
        self.dropout = nn.Dropout(dec_config.hidden_dropout_prob)
        # self.mealPooling_module = nn.AvgPool1d((enc_config.max_position_embeddings, 1))  # utilize pooling out instead of mean pooling
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.mask_token_ids = tokenizer.mask_token_ids
        self.cor_token_id = tokenizer.convert_tokens_to_ids("<COR>")
        self.pln_token_id = tokenizer.convert_tokens_to_ids("<PLN>")
        assert dec_config.hidden_size == 768, "the hidden state must be the same with sentence bert"
        self.max_discriminator = Max_Discriminator(dec_config.hidden_size, initrange=0.1)
        # self.init_weights()
        self.args = args
        self.sim_alpha = args.sim_alpha
        
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def tie_cls_weight(self):
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight

    def forward(self, batch, prev_output_tokens):
        input_ids = batch['src_idx']
        tgt_ids_first = batch['tgt_idx_first']
        tgt_ids_second = batch['tgt_idx_second']  # 
        tgt_attention_mask = batch['tgt_attention_mask']
        attention_mask = batch['masks']
        loss_mask = batch['loss_masks']
        label_ids = batch['tgt_idx'] if 'tgt_idx' in batch else None
        sn_pos = batch['sn_pos'] if 'sn_pos' in batch else None
        
        # step 1: calculate the KL loss
        enc_out = self.encoder(input_ids)  
        ipt_features = self.mean_pooling(
            enc_out, attention_mask
        )
        ipt_features_expend = torch.stack(
            [ipt_features] * torch.sum(sn_pos), dim=1) # expend to sn_pos
        ipt_features_expend = ipt_features_expend.squeeze()  # remove the dimension which equals to 1

        dec_out = self.decoder(
            input_ids=tgt_ids_first, 
            attention_mask=tgt_attention_mask,
            encoder_hidden_states=enc_out.last_hidden_state, 
        )
        sen_features = dec_out.index_select(sn_pos, dim=1)  # extract plan hidden states
        loss_sim = self.multul_info(ipt_features_expend, sen_features)
        
        # step2: generation according to the sen_features


    def multul_info(self, dis1, dis2):
        dis_rnd = torch.cat((dis2[1:], dis2[0].unsqueeze(0)), dim=0)
        Ej = -F.softplus(-self.max_discriminator(dis1, dis2))
        Em = F.softplus(self.max_discriminator(dis1, dis_rnd))
        return Em - Ej


   


import torch
from typing import List, Optional, Tuple, Union


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    # fill the padding position with -inf
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _make_causal_mask(input_ids_shape, dtype, past_key_values_length=0, sn_position=None, _add_special_token_mask=False):
        """
        Make causal mask used for span mask self-attention.
        The index of the sentence begins from 1 not 0.
        """ 
        bsz, tgt_len = input_ids_shape
        normal_mask = torch.zeros((bsz, tgt_len, tgt_len))
        span_mask = torch.zeros((bsz, tgt_len, tgt_len))
        special_token_mask = torch.zeros((bsz, tgt_len))  # first mask then expend
        mask_cond = torch.arange(normal_mask.size(-1))

        if sn_position is not None:
            matrix1 = torch.stack([sn_position] * tgt_len, 1)
            matrix2 = matrix1.permute(0, 2, 1) + 1
            span_mask.masked_fill_(matrix1 < matrix2, 1)
            span_mask.to(dtype)
        
        if _add_special_token:
            # mask according to the special tokens
            x_ids, y_ids = torch.arange(0, bsz)[:, None], sn_position - 1
            special_token_mask[x_ids, y_ids] = 1
            special_token_mask = torch.stack([special_token_mask] * tgt_len, 1)

        normal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(normal_mask.size(-1), 1), 1)
        normal_mask = normal_mask.to(dtype)
        to_mask = normal_mask + span_mask + special_token_mask
        '''
        if mask with 1, non mask
        if mask with 0, mask
        '''
        # return to_mask.where(to_mask<1, torch.tensor(1)) # saint check
        to_mask = torch.where(to_mask>=1, 0, torch.tensor(torch.finfo(dtype).min))  # convert to 0 and -inf 
        if past_key_values_length > 0:
            to_mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), to_mask], dim=-1)
        return to_mask[:, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)



def _prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length, sn_position=None, _add_special_token_mask=False):
    # create casual mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length, 
            sn_position=sn_position, _add_special_token_mask=_add_special_token_mask
        ).to(inputs_embeds.device)
    
    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

        return combined_attention_mask

def _mask_casual_sential_attention_mask(special_pos, max_seq_len):
    res = []
    prev = 0
    for i in special_pos:
        res += [i] * (i - prev)
        prev = i
    assert prev == max_seq_len and len(res) == max_seq_len, \
        "position should utill meets max_seq_len, check if the last token is <PLN>"
    return res


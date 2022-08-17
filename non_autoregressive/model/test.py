from numpy import dtype
from utils import (
    _make_causal_mask,
    _mask_casual_sential_attention_mask
)
import torch
import numpy as np
import random

def int_random(a, b, n, max_len) :
    a_list = []
    x = random.randint(1, n)
    while len(a_list) < x - 1 :
        d_int = random.randint(a, b)
        if(d_int not in a_list) :
            a_list.append(d_int)
        else:
            pass
    a_list.append(max_len)
    return sorted(a_list)

bsz, tgt_len = 2, 10
input_ids_shape = (bsz, tgt_len)
dtype = torch.float
list_pre = [int_random(1, tgt_len, tgt_len, tgt_len) for i in range(bsz)]
list_pre = [_mask_casual_sential_attention_mask(item, tgt_len) for item in list_pre]
special_pos = torch.tensor(list_pre, dtype=torch.long)
res = _make_causal_mask(input_ids_shape, dtype, sn_position=special_pos, _add_special_token=True)

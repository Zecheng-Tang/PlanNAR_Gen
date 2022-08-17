from matplotlib.pyplot import axis
import torch
import torch.nn.functional as F
import sys

def gather_nd(x, indices):
    newshape = list(indices.shape[:-1] + x.shape[indices.shape[-1]:]) + [1]
    indices = indices.view(-1, indices.shape[-1]).tolist()
    out = torch.cat([torch.tensor([x.__getitem__(tuple(i))]) for i in indices]).reshape(tuple(newshape))
    return out

def top_p_logits(logits, p):
    # Nucleus sampling
    batch, _ = logits.size()
    sorted_logits, _ = torch.sort(logits, descending=True, axis=-1)
    cumulative_probs, _ = torch.cumsum(F.softmax(sorted_logits), axis=-1)
    cumulative_position = torch.sum((cumulative_probs <= p).to(torch.int32), axis=-1) - 1
    indices = torch.stack([
        torch.arange(0, batch).to(device),
        # number of indices to include
        torch.max(cumulative_position, torch.zeros([batch], dtype=cumulative_position.type).to(device))
    ], axis=-1)
    # return the min values
    min_values = gather_nd(sorted_logits, indices).to(device)
    return torch.where(
        logits < min_values,
        torch.ones_like(logits) * 1e-10,
        logits
    )





device = sys.argv[1]
tokenizer = None
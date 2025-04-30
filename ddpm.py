import torch
import math


# Sinusoidal positional embedding
def positional_embedding(x, emb_dim):
    """
    x: input tensor of shape (number of points, 1)
    emb_dim: int, embedding dimension
    """
    if emb_dim % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dim (got dim={:d})".format(emb_dim))
    pe = torch.zeros(x.shape[0], emb_dim)
    div_term = torch.exp(torch.arange(0, emb_dim, 2, dtype=torch.float) * -(math.log(10000.0) / emb_dim))
    pe[:, 0::2] = torch.sin(x * div_term)
    pe[:, 1::2] = torch.cos(x * div_term)
    return pe


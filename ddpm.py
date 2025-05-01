import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

import datasets
import argparse


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


class MLP(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, emb_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PointDiffusionModel(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.input_dim = 3 * emb_dim
        self.emb_dim = emb_dim
        self.mlp = MLP(self.input_dim, emb_dim)

    def forward(self, x, t):
        return self.mlp(x)


def main(args):

    # Prepare the dataset
    dataset = datasets.get_dataset(args.dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # x 하나씩 꺼내서 밑에 넣어주기

    # Model
    t = torch.randint(0, args.num_timesteps, (1,))
    model = PointDiffusionModel(emb_dim=args.emb_dim)
    
    # input
    x = torch.cat([positional_embedding(t, emb_dim=args.emb_dim), 
                    positional_embedding(x[:, 0], emb_dim=args.emb_dim), 
                    positional_embedding(x[:, 1], emb_dim=args.emb_dim)])

    # Add noise to the input
    noise = torch.randn_like(x)
    x_noisy = x + noise
    # Pass through the MLP
    x_pred = model(x_noisy)
    
    loss = F.mse_loss(x_pred, x)
    loss.backward()
    




    # Noise Scheduler


    # Optimizer


    # Training Loop


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point diffusion model")
    parser.add_argument("--dataset", type=str, default="dino", help="Dataset to use")
    parser.add_argument("--n", type=int, default=100, help="Number of samples")
    parser.add_argument("--emb_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num_timesteps", type=int, default=50, help="Number of time steps")
    args = parser.parse_args()

    main(args)

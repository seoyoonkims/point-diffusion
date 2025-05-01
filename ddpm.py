import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

import datasets
import argparse
from datetime import datetime

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
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

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
        self.mlp = MLP(self.input_dim, output_dim=2) # output_dim=2 for 2D points

    def forward(self, x, t):
        # positional embedding
        x_emb = torch.cat([positional_embedding(t, emb_dim=self.emb_dim), 
                        positional_embedding(x[:, 0], emb_dim=self.emb_dim), 
                        positional_embedding(x[:, 1], emb_dim=self.emb_dim)], dim=1)
        return self.mlp(x_emb)


class beta_scheduler(nn.Module):
    def __init__(self, num_timesteps):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps)


class NoiseScheduler(nn.Module):
    def __init__(self, num_timesteps):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.betas = beta_scheduler(self.num_timesteps).betas
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def get_alpha_t(self, t):
        return self.alphas_cumprod.gather(0, t)      # gather: 위치마다 다른 인덱싱
    
    def forward(self, x, t, noise):
        alpha_t = self.get_alpha_t(t).unsqueeze(1)
        return x * torch.sqrt(alpha_t) + noise * torch.sqrt(1 - alpha_t)


def main(args):
    # Prepare the dataset
    dataset = datasets.get_dataset(args.dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = PointDiffusionModel(emb_dim=args.emb_dim)
    noise_scheduler = NoiseScheduler(args.num_timesteps)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.num_iteration):

        # Training loop
        model.train()
        total_loss = 0        
        for i, data in enumerate(dataloader):
            # raw data
            x = data[0]
            t = torch.randint(0, args.num_timesteps, (x.size(0),), device=x.device)
            
            # add nosie
            noise = torch.randn_like(x)
            x_noisy = noise_scheduler(x, t, noise)
            noise_pred = model(x_noisy, t)

            # loss
            loss = F.mse_loss(noise_pred, noise)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")


        # Evaluation
        model.eval()
        with torch.no_grad():
            pass
    
    # Storing the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"point_diffusion_model_{timestamp}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point diffusion model")
    parser.add_argument("--dataset", type=str, default="dino", help="Dataset to use")
    parser.add_argument("--n", type=int, default=100, help="Number of samples")
    parser.add_argument("--emb_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num_timesteps", type=int, default=50, help="Number of time steps")
    parser.add_argument("--num_iteration", type=int, default=200, help="Number of iterations")
    args = parser.parse_args()

    main(args)

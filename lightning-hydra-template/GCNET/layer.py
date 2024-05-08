import torch
import torch.nn as nn
from GCNET.attention import *
from torch_geometric.nn import global_max_pool, global_mean_pool


class Emb(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.x = nn.Parameter(torch.normal(0, 1, size=(embedding_size,)))
        self.a = nn.Parameter(torch.normal(0, 1, size=(embedding_size,)))
        self.b = nn.Parameter(torch.normal(0, 1, size=(embedding_size,)))

    def forward(self, degree):
        x = self.x * degree.unsqueeze(1)
        x = self.a * torch.cos(x) + self.b
        return x


class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, n_head, n_layers, drop_prob):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(embedding_size, hidden_size, n_head, drop_prob) for _ in range(n_layers)])

    def forward(self, x, edge_index, batch, latent):
        scores = []
        for layer in self.layers:
            x, score = layer(x, edge_index, batch, latent)
            if latent:
                scores.append(score)
        x = global_mean_pool(x, batch)
        return (x, None) if not latent else (x, scores)


class LastLayer(nn.Module):
    def __init__(self, embedding_size, last_layer_dim, drop_prob, is_prob):
        super().__init__()
        self.is_prob = is_prob
        self.log_soft = nn.Softmax(dim=-1)

        dim = [embedding_size] + last_layer_dim
        last_layer = []
        for idx in range(len(dim) - 2):
            last_layer += [nn.Linear(dim[idx], dim[idx + 1]), nn.BatchNorm1d(dim[idx + 1]), nn.GELU(), nn.Dropout(drop_prob)]
        last_layer.append(nn.Linear(dim[-2], dim[-1]))
        self.lastlayer = nn.Sequential(*last_layer)

    def forward(self, x):
        for _, layer in enumerate(self.lastlayer):
            x = layer(x)
        return self.log_soft(x) if self.is_prob else x.squeeze()

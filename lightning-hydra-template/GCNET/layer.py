import torch
import torch.nn as nn
from GCNET.attention import *


class EmbeddingLayer(nn.Module):
    def __init__(self, node_number, embedding_size, embed_layer):
        super().__init__()
        self.embed = nn.Embedding(node_number, embedding_size, padding_idx=0)
        self.embed_layer = embed_layer

    def forward(self, x):
        degree = torch.sum(x, dim=1)
        embed = self.embed(degree.type(torch.long)).permute(0, 2, 1)
        adj = torch.where(degree > 0, 1 / torch.pow(degree, 0.5), 0)
        adj = x * adj.reshape([*degree.shape, 1])

        emb = [embed]
        for _ in range(self.embed_layer):
            emb.append(emb[-1] @ adj)
        emb = torch.stack(emb).mean(axis=0)
        return emb, adj


class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, n_head, n_layers, drop_prob):
        super().__init__()
        self.n_head = n_head
        self.layers = nn.ModuleList([EncoderLayer(embedding_size, hidden_size, n_head, drop_prob) for _ in range(n_layers)])

    def forward(self, x, adj):
        batch_size, _, node_size = x.shape
        adj = torch.unsqueeze(adj, 1).expand(batch_size, self.n_head, node_size, node_size)
        for layer in self.layers:
            x = layer(x, adj)
        x = torch.max(x, dim=-1).values
        return x


class LastLayer(nn.Module):
    def __init__(self, embedding_size, last_layer_dim, drop_prob, is_prob):
        super().__init__()
        self.is_prob = is_prob
        self.log_soft = nn.LogSoftmax(dim=1)

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

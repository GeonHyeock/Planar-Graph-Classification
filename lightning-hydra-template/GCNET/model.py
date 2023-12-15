import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import networkx as nx


class EmbeddingLayer(nn.Module):
    def __init__(self, node_number, embedding_size):
        super().__init__()
        self.embed = nn.Embedding(node_number, embedding_size, padding_idx=0)
        nn.init.xavier_uniform_(self.embed.weight[1:])

    def forward(self, x, device):
        degree = torch.sum(x, dim=1)
        embed = self.embed(degree.type(torch.long)).permute(0, 2, 1)
        adj = x + torch.eye(*x.size()[1:]).to(device)
        return adj, embed


class ConvBlock(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        es = embedding_size
        self.block = nn.Sequential(
            nn.Conv1d(es, es * 2, 1),
            nn.BatchNorm1d(es * 2),
            nn.ReLU(),
            nn.Conv1d(es * 2, es * 2, 1),
            nn.BatchNorm1d(es * 2),
            nn.ReLU(),
            nn.Conv1d(es * 2, es, 1),
            nn.BatchNorm1d(es),
            nn.ReLU(),
        )

        for layer in self.block:
            if isinstance(layer, nn.Conv1d):
                nn.init.xavier_normal(layer.weight)

    def forward(self, x):
        return self.block(x)


class BLOCKS(nn.Module):
    def __init__(self, embedding_size, block_layer):
        super().__init__()
        self.layers = nn.ModuleList()
        for idx in range(block_layer):
            self.layers.append(ConvBlock(embedding_size))

    def forward(self, adj, x):
        for idx, layer in enumerate(self.layers):
            x = x + layer(x) @ (adj / (idx + 1))
        return torch.max(x, dim=2).values


class GCnet(nn.Module):
    def __init__(
        self,
        node_number=50,
        embedding_size=256,
        block_layer=3,
        classes=2,
    ):
        super().__init__()
        self.embedding = EmbeddingLayer(node_number, embedding_size)
        self.convblocks = BLOCKS(embedding_size, block_layer)
        self.last_layer = nn.Linear(embedding_size, classes)
        self.log_soft = nn.LogSoftmax(dim=1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        adj, x = self.embedding(x, self.device)
        x = self.Tnet(adj, x)
        return self.log_soft(self.last_layer(x))


if __name__ == "__main__":
    data1 = pd.read_csv("/home/user/project/data/version_001/00000000.csv").values
    data1 = nx.to_numpy_array(nx.from_edgelist(data1))

    data2 = pd.read_csv("/home/user/project/data/version_001/00000001.csv").values
    data2 = nx.to_numpy_array(nx.from_edgelist(data2))

    data1 = np.pad(data1, ((0, 32), (0, 32)))

    net = GCnet()
    x = torch.stack((torch.from_numpy(data1), torch.from_numpy(data2)), dim=0)
    output = net(x)

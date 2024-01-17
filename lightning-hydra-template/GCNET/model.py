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

    def forward(self, x):
        degree = torch.sum(x, dim=1)
        embed = self.embed(degree.type(torch.long)).permute(0, 2, 1)
        adj = x
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
                nn.init.xavier_normal_(layer.weight)

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


class LastLayer(nn.Module):
    def __init__(self, embedding_size, layer, classes):
        super().__init__()
        layer -= 1
        self.lastlayer = nn.ModuleList()
        for idx in range(layer):
            self.lastlayer.append(
                nn.Sequential(
                    nn.Linear(
                        embedding_size // (2**idx), embedding_size // (2 ** (idx + 1))
                    ),
                    nn.BatchNorm1d(embedding_size // (2 ** (idx + 1))),
                    nn.GELU(),
                    nn.Dropout(p=0.2),
                )
            )
        self.lastlayer.append(nn.Linear(embedding_size // (2 ** (layer)), classes))

    def forward(self, x):
        for idx, layer in enumerate(self.lastlayer):
            x = layer(x)
        return x


class GCnet(nn.Module):
    def __init__(
        self,
        node_number=50,
        embedding_size=256,
        block_layer=3,
        last_layer=3,
        classes=2,
    ):
        super().__init__()
        self.embedding = EmbeddingLayer(node_number, embedding_size)
        self.convblocks = BLOCKS(embedding_size, block_layer)
        self.last_layer = LastLayer(embedding_size, last_layer, classes)
        self.log_soft = nn.LogSoftmax(dim=1)
        self.classes = classes

    def forward(self, x, latent=False):
        adj, x = self.embedding(x)
        x = self.convblocks(adj, x)
        if latent:
            return x
        return self.log_soft(self.last_layer(x))


if __name__ == "__main__":
    from itertools import permutations

    def errors(output):
        errors = []
        for i in range(len(output)):
            for j in range(i + 1, len(output)):
                error = torch.pow(((output[i] - output[j]) ** 2).sum(), 0.5)
                errors.append(error)
        errors = torch.stack(errors)
        return errors.max().cpu(), errors.mean().cpu(), errors.min().cpu()

    def invariant_test(G_node):
        print("Isomorphism_error")
        batch, n = [], max(sum(G_node, [])) + 1
        for permu in permutations(range(n), n):
            data = np.zeros((n, n))
            for i, j in G_node:
                data[permu[i], permu[j]] = 1
                data[permu[j], permu[i]] = 1
            batch.append(torch.from_numpy(data))
        batch = torch.stack(batch).type(torch.float32).cuda()

        with torch.no_grad():
            net = GCnet().cuda()
            output = net(batch, latent=True)
            max_error, mean_error, min_error = errors(output)
            print(
                f"max error : {max_error:.8f}, mean error : {mean_error:.8f} min error : {min_error:.8f}"
            )

    def initial_error(G_node):
        print("base_error")
        n = max(sum(G_node, [])) + 1
        d1 = nx.to_numpy_array(nx.cycle_graph(n))
        d2 = nx.to_numpy_array(nx.star_graph(n - 1))
        d3 = nx.to_numpy_array(nx.complete_graph(n))
        d4 = nx.to_numpy_array(nx.path_graph(n))
        d5 = nx.to_numpy_array(nx.Graph(G_node))
        data = torch.tensor(np.stack([d1, d2, d3, d4, d5]), dtype=torch.float32).cuda()

        with torch.no_grad():
            net = GCnet().cuda()
            output = net(data, latent=True)
            max_error, mean_error, min_error = errors(output)
            print(
                f"max error : {max_error:.8f}, mean error : {mean_error:.8f} min error : {min_error:.8f}"
            )

    G_node = [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3], [3, 4]]
    invariant_test(G_node)
    initial_error(G_node)

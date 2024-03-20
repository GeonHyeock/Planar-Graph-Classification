import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from GCNET.layer import *


class GCnet(nn.Module):
    def __init__(
        self,
        node_number=50,
        embedding_size=512,
        embed_layer=2,
        n_layers=4,
        hidden_size=256,
        n_head=8,
        drop_prob=0.1,
        last_layer_dim=[256, 1],
        is_prob=False,
    ):
        super().__init__()
        self.embedding = EmbeddingLayer(node_number, embedding_size, embed_layer)
        self.encoder = Encoder(embedding_size, hidden_size, n_head, n_layers, drop_prob)
        self.last = LastLayer(embedding_size, last_layer_dim, drop_prob, is_prob)

    def forward(self, x, latent=False):
        emb, adj = self.embedding(x)
        x = self.encoder(emb, adj)
        if latent:
            return x
        return self.last(x)


if __name__ == "__main__":
    from itertools import permutations

    def cos_sims(output):
        cos_sims = []
        for i in range(len(output)):
            for j in range(i + 1, len(output)):
                cos_sim = torch.dot(output[i], output[j]) / torch.norm(output[i]) / torch.norm(output[j])
                cos_sims.append(cos_sim)
        cos_sims = torch.stack(cos_sims)
        return cos_sims.max().cpu(), cos_sims.mean().cpu(), cos_sims.min().cpu()

    def invariant_test(G_node):
        print("Isomorphism_cos_sim")
        batch, n = [], max(sum(G_node, [])) + 1
        for permu in permutations(range(n), n):
            data = np.zeros((n, n))
            for i, j in G_node:
                data[permu[i], permu[j]] = 1
                data[permu[j], permu[i]] = 1
            batch.append(torch.from_numpy(data))
        batch = torch.stack(batch).type(torch.float32).cuda()

        output = net(batch, latent=True)
        max_cos_sim, mean_cos_sim, min_cos_sim = cos_sims(output)
        print(f"max cos_sim : {max_cos_sim:.8f}, mean cos_sim : {mean_cos_sim:.8f} min cos_sim : {min_cos_sim:.8f}")

    def initial_cos_sim(G_node):
        print("base_cos_sim")
        n = max(sum(G_node, [])) + 1
        d1 = nx.to_numpy_array(nx.cycle_graph(n))
        d2 = nx.to_numpy_array(nx.star_graph(n - 1))
        d3 = nx.to_numpy_array(nx.complete_graph(n))
        d4 = nx.to_numpy_array(nx.path_graph(n))
        d5 = nx.to_numpy_array(nx.Graph(G_node))
        data = torch.tensor(np.stack([d1, d2, d3, d4, d5]), dtype=torch.float32).cuda()

        output = net(data, latent=True)
        max_cos_sim, mean_cos_sim, min_cos_sim = cos_sims(output)
        print(f"max cos_sim : {max_cos_sim:.8f}, mean cos_sim : {mean_cos_sim:.8f} min cos_sim : {min_cos_sim:.8f}")

    net = GCnet().cuda()
    net.eval()
    G_node = [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3], [3, 4]]
    invariant_test(G_node)
    initial_cos_sim(G_node)

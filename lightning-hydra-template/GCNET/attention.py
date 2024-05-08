import torch
import torch.nn as nn
from torch_geometric.nn.aggr import MeanAggregation


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.agg = MeanAggregation()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, edge_index, latent):
        A, B = edge_index
        num_head = q.shape[1]
        score = q[A] @ k.transpose(-1, -2)[B]
        score = torch.stack([self.agg(score[:, idx, :], A) for idx in range(num_head)], dim=1)
        score = self.softmax(score)
        v = score @ v
        return (v, None) if not latent else (v, score)


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_size, n_head):
        super().__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(embedding_size, embedding_size)
        self.w_k = nn.Linear(embedding_size, embedding_size)
        self.w_v = nn.Linear(embedding_size, embedding_size)
        self.w_concat = nn.Linear(embedding_size, embedding_size, 1)

    def forward(self, q, k, v, edge_index, latent):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, score = self.attention(q, k, v, edge_index, latent)
        out = self.w_concat(self.concat(out))
        return out, score

    def split(self, tensor):
        node_size, embedding_size = tensor.size()
        d_tensor = embedding_size // self.n_head
        tensor = tensor.view(node_size, self.n_head, d_tensor)
        return tensor

    def concat(self, tensor):
        node_size, n_head, d_tensor = tensor.size()
        embedding_size = n_head * d_tensor
        tensor = tensor.view(node_size, embedding_size)
        return tensor


class FeedForward(nn.Module):
    def __init__(self, embedding_size, hidden_size, drop_prob):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_size, embedding_size),
        )

    def forward(self, x):
        return self.block(x)


class EncoderLayer(nn.Module):
    def __init__(self, embedding_size, hidden_size, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embedding_size, n_head)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm1 = nn.BatchNorm1d(embedding_size)

        self.ffn = FeedForward(embedding_size, hidden_size, drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.norm2 = nn.BatchNorm1d(embedding_size)

    def forward(self, x, edge_index, batch, latent):
        _x = x
        x, score = self.attention(x, x, x, edge_index, latent)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x, score

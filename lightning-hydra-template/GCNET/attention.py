import torch
import torch.nn as nn


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, adj):
        k_t = k.transpose(2, 3)
        score = ((q @ k_t) * adj).masked_fill(adj == 0, float("-inf"))
        score = torch.nan_to_num(self.softmax(score))
        v = score @ v
        return v


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_size, n_head):
        super().__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Conv1d(embedding_size, embedding_size, 1)
        self.w_k = nn.Conv1d(embedding_size, embedding_size, 1)
        self.w_v = nn.Conv1d(embedding_size, embedding_size, 1)
        self.w_concat = nn.Conv1d(embedding_size, embedding_size, 1)

    def forward(self, q, k, v, adj):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out = self.attention(q, k, v, adj)
        out = self.w_concat(self.concat(out))
        return out

    def split(self, tensor):
        batch_size, embedding_size, node_size = tensor.size()
        d_tensor = embedding_size // self.n_head
        tensor = tensor.view(batch_size, self.n_head, d_tensor, node_size).transpose(2, 3)
        return tensor

    def concat(self, tensor):
        batch_size, head, node_size, d_tensor = tensor.size()
        embedding_size = head * d_tensor
        tensor = tensor.transpose(2, 3).contiguous().view(batch_size, embedding_size, node_size)
        return tensor


class FeedForward(nn.Module):
    def __init__(self, embedding_size, hidden_size, drop_prob):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(embedding_size, hidden_size, 1),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Conv1d(hidden_size, embedding_size, 1),
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

    def forward(self, x, adj):
        _x = x
        x = self.attention(q=x, k=x, v=x, adj=adj)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

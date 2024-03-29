import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.pygcn.layers import GraphConvolution
from GCNET.layer import LastLayer


class GCN(nn.Module):
    def __init__(self, node_number, embedding_size, nfeat, nhid, dropout, last_layer_dim, is_prob):
        super(GCN, self).__init__()

        self.emb = nn.Embedding(node_number, embedding_size, padding_idx=0)
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout
        self.last_layer = LastLayer(embedding_size, last_layer_dim, dropout, is_prob)

    def forward(self, adj):
        x = self.emb(torch.sum(adj, dim=-1).type(torch.long))
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = torch.max(x, dim=1).values
        return self.last_layer(x)

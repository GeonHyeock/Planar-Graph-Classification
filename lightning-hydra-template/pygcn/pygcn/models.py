import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, node_number, embedding_size, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.emb = nn.Embedding(node_number, embedding_size)
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, adj):
        x = self.emb(torch.sum(adj, dim=-1).type(torch.long))
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = torch.mean(x, dim=1)
        return self.softmax(x)

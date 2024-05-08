#
# Copyright (C)  2020  University of Pisa
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Embedding
from torch.nn import Sequential, Linear, ReLU, Softmax
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool


class GIN(torch.nn.Module):

    def __init__(
        self,
        node_number,
        embedding_size,
        dim_target,
        hidden_units,
        train_eps,
        aggregation,
        dropout,
    ):
        super(GIN, self).__init__()

        self.dropout = dropout
        self.emb = Embedding(node_number, embedding_size, padding_idx=0)
        self.embeddings_dim = [hidden_units[0]] + hidden_units
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []

        if aggregation == "sum":
            self.pooling = global_add_pool
        elif aggregation == "mean":
            self.pooling = global_mean_pool

        for layer, out_emb_dim in enumerate(self.embeddings_dim):

            if layer == 0:
                self.first_h = Sequential(
                    Linear(embedding_size, out_emb_dim),
                    BatchNorm1d(out_emb_dim),
                    ReLU(),
                    Linear(out_emb_dim, out_emb_dim),
                    BatchNorm1d(out_emb_dim),
                    ReLU(),
                )
                self.linears.append(Linear(out_emb_dim, dim_target))
            else:
                input_emb_dim = self.embeddings_dim[layer - 1]
                self.nns.append(
                    Sequential(
                        Linear(input_emb_dim, out_emb_dim),
                        BatchNorm1d(out_emb_dim),
                        ReLU(),
                        Linear(out_emb_dim, out_emb_dim),
                        BatchNorm1d(out_emb_dim),
                        ReLU(),
                    )
                )
                self.convs.append(GINConv(self.nns[-1], train_eps=train_eps))  # Eq. 4.2

                self.linears.append(Linear(out_emb_dim, dim_target))

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        x, edge_index, batch = x
        x = self.emb(x)

        out = 0
        for layer in range(self.no_layers):
            if layer == 0:
                x = self.first_h(x)
                out += F.dropout(self.pooling(self.linears[layer](x), batch), p=self.dropout)
            else:
                # Layer l ("convolution" layer)
                x = self.convs[layer - 1](x, edge_index)
                out += F.dropout(self.linears[layer](self.pooling(x, batch)), p=self.dropout, training=self.training)

        return self.softmax(out)

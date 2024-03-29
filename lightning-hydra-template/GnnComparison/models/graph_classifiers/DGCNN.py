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
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing, global_sort_pool
from torch_geometric.utils import add_self_loops, degree, dense_to_sparse


class DGCNN(nn.Module):
    """
    Uses fixed architecture
    """

    def __init__(self, node_number, embedding_size, dim_features, num_layers, dense_dim, dim_target, k):
        super(DGCNN, self).__init__()

        self.k = k
        self.emb = nn.Embedding(node_number, embedding_size)
        self.convs = []
        for layer in range(num_layers):
            input_dim = embedding_size if layer == 0 else dim_features
            self.convs.append(DGCNNConv(input_dim, dim_features))
        self.total_latent_dim = num_layers * dim_features

        # Add last embedding
        self.convs.append(DGCNNConv(dim_features, 1))
        self.total_latent_dim += 1

        self.convs = nn.ModuleList(self.convs)

        # should we leave this fixed?
        self.conv1d_params1 = nn.Conv1d(1, 16, self.total_latent_dim, self.total_latent_dim)
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(16, 32, 5, 1)

        dense_dim = int((self.k - 2) / 2 + 1)
        self.input_dense_dim = (dense_dim - 5 + 1) * 32

        self.hidden_dense_dim = dense_dim
        self.dense_layer = nn.Sequential(
            nn.Linear(self.input_dense_dim, self.hidden_dense_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.hidden_dense_dim, dim_target),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, data):
        if data.shape[0] > 1:
            x = torch.stack([self.online_forward(d) for d in data]).squeeze()
        else:
            x = self.online_forward(data.squeeze())
        return self.softmax(x)

    def online_forward(self, data, batch=None):
        # Implement Equation 4.2 of the paper i.e. concat all layers' graph representations and apply linear model
        # note: this can be decomposed in one smaller linear model per layer
        x = self.emb(torch.sum(data, dim=-1).type(torch.long))
        edge_index, _ = dense_to_sparse(data)

        hidden_repres = []

        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            hidden_repres.append(x)

        # apply sortpool
        x_to_sortpool = torch.cat(hidden_repres, dim=1)
        x_1d = global_sort_pool(x_to_sortpool, batch, self.k)  # in the code the authors sort the last channel only

        # apply 1D convolutional layers
        x_1d = torch.unsqueeze(x_1d, dim=1)
        conv1d_res = F.relu(self.conv1d_params1(x_1d))
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = F.relu(self.conv1d_params2(conv1d_res))
        conv1d_res = conv1d_res.reshape(conv1d_res.shape[0], -1)

        # apply dense layer
        out_dense = self.dense_layer(conv1d_res)
        return out_dense


class DGCNNConv(MessagePassing):
    """
    Extended from tuorial on GCNs of Pytorch Geometrics
    """

    def __init__(self, in_channels, out_channels):
        super(DGCNNConv, self).__init__(aggr="add")  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        src, dst = edge_index  # we assume source_to_target message passing
        deg = degree(src, size[0], dtype=x_j.dtype)
        deg = deg.pow(-1)
        norm = deg[dst]

        return norm.view(-1, 1) * x_j  # broadcasting the normalization term to all out_channels === hidden features

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)

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
from torch import nn
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_geometric.nn import SAGEConv, global_max_pool


class GraphSAGE(nn.Module):
    def __init__(self, node_number, embedding_size, dim_features, dim_target, aggregation, num_layers):
        super().__init__()

        self.aggregation = aggregation
        self.emb = nn.Embedding(node_number, embedding_size, padding_idx=0)

        if self.aggregation == "max":
            self.fc_max = nn.Linear(embedding_size, embedding_size)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = embedding_size if i == 0 else dim_features

            # Overwrite aggregation method (default is set to mean
            conv = SAGEConv(dim_input, dim_features, aggr=self.aggregation)

            self.layers.append(conv)

        # For graph classification
        self.fc1 = nn.Linear(num_layers * dim_features, dim_features)
        self.fc2 = nn.Linear(dim_features, dim_target)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x, edge_index, batch = x
        x = self.emb(x)

        x_all = []

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.aggregation == "max":
                x = torch.relu(self.fc_max(x))
            x_all.append(x)

        x = torch.cat(x_all, dim=1)
        x = global_max_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

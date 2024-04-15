from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse
import torch
import networkx as nx
import numpy as np
import pandas as pd
import os, re


class GraphDataset(Dataset):
    def __init__(self, data_dir, data_version, data_type):
        self.data_dir = data_dir[:-5]
        self.data_type = data_type
        self.r = re.compile("version_...")
        if isinstance(data_version, str):
            self.data = pd.read_csv(os.path.join(data_dir, data_version, "label.csv"))
            self.data = self.data[self.data.type == data_type].reset_index(drop=True)
        else:
            self.data = []
            for dv in data_version:
                data = pd.read_csv(os.path.join(data_dir, dv, "label.csv"))
                self.data.append(data[data.type == data_type])
            self.data = pd.concat(self.data).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = dict(self.data.iloc[idx])
        G = nx.read_adjlist(os.path.join(self.data_dir, d["data_path"]))
        version = self.r.search(d["data_path"]).group()
        adj = torch.Tensor(nx.to_numpy_array(G))
        edge_index, _ = dense_to_sparse(adj)
        degree = adj.sum(dim=-1)

        return {
            "version": version,
            "edge": edge_index,
            "degree": degree,
            "label": d["label_name"],
        }


def my_collate_fn(batch):
    b = pd.DataFrame(batch)
    V, E, D, B, cumsum = [], [], [], [], 0
    for idx in range(len(b)):
        V.append(b.iloc[idx].version)
        E.append(b.iloc[idx].edge + cumsum)
        D.append(b.iloc[idx].degree)
        B.extend([idx] * len(b.iloc[idx].degree))
        cumsum += len(b.iloc[idx].degree)

    return {
        "version": V,
        "edge": torch.cat(E, dim=1),
        "degree": torch.cat(D).type(torch.long),
        "batch": torch.tensor(B).type(torch.long),
        "label": torch.tensor(b.label).type(torch.long),
    }


def my_padding(x, max_size):
    padding_size = max_size - len(x)
    x = np.pad(x, ((0, padding_size), (0, padding_size)))
    return torch.from_numpy(x)

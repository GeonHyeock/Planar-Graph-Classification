from torch.utils.data import Dataset
import torch
import networkx as nx
import numpy as np
import pandas as pd
import os


class GraphDataset(Dataset):
    def __init__(self, data_version, data_type):
        self.data_folder = data_version[: data_version.find("data/version")]
        self.data = pd.read_csv(os.path.join(data_version, "label.csv"))
        self.data = self.data[self.data.type == data_type].reset_index(drop=True)

        label_dict = pd.read_csv(os.path.join(data_version, "label_dict.csv"))
        self.label_dict = {
            c: l for c, l in zip(label_dict.label_name, label_dict.label)
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = dict(self.data.iloc[idx])
        G = pd.read_csv(os.path.join(self.data_folder, d["data_path"])).values
        G = nx.to_numpy_array(nx.from_edgelist(G))
        d.update({"graph": G})
        d["label"] = self.label_dict[d["label_name"]]
        return d


def my_collate_fn(batch):
    b = pd.DataFrame(batch)
    max_size = b.graph.apply(len).max()
    b.graph = b.graph.apply(lambda x: my_padding(x, max_size))
    b = b.to_dict(orient="list")

    b["graph"] = torch.stack(b["graph"]).type(torch.float32)
    b["label"] = torch.tensor(b["label"]).type(torch.LongTensor)
    return b


def my_padding(x, max_size):
    padding_size = max_size - len(x)
    x = np.pad(x, ((0, padding_size), (0, padding_size)))
    return torch.from_numpy(x)

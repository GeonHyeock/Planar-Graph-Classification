import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import hydra
import time
import torch
import networkx as nx
import lightning as L
import pandas as pd
from tqdm import tqdm
from typing import Optional
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
from collections import defaultdict
from src.utils import extras


def algorithm(dataloader):
    infer_time = 0
    for batch in dataloader:
        G = nx.from_edgelist(batch["edge"].T)
        start = time.perf_counter()
        nx.is_planar(G)
        end = time.perf_counter()
        infer_time += end - start
    return infer_time


def cpu_model(dataloader, net):
    infer_time = 0
    net = net.to("cpu")
    with torch.no_grad():
        for batch in dataloader:
            edge, degree, batch = batch["edge"], batch["degree"], batch["batch"]
            start = time.perf_counter()
            net((degree, edge, batch)).max(dim=-1).indices
            end = time.perf_counter()
            infer_time += end - start
    return infer_time


def gpu_model(dataloader, net):
    infer_time = 0
    net = net.to("cuda")
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for batch in dataloader:
            edge, degree, batch = batch["edge"].to("cuda"), batch["degree"].to("cuda"), batch["batch"].to("cuda")
            starter.record()
            net((degree, edge, batch)).max(dim=-1).indices
            ender.record()
            torch.cuda.synchronize()
            infer_time += starter.elapsed_time(ender) * 1e-3
    return infer_time


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    cfg.get("model").DataVersion = cfg.get("data").data_version
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    net = model.net.eval()
    dataloader = datamodule.test_dataloader()

    df = defaultdict(list)
    for _ in tqdm(range(10)):
        df["algorithm"].append(algorithm(dataloader))
        df["cpu_model"].append(cpu_model(dataloader, net))
        df["gpu_model"].append(gpu_model(dataloader, net))
    df = pd.DataFrame(df)
    model_name = cfg.model.net._target_.split(".")[-1]
    df.to_csv(f"../data/{model_name}_inference_time.csv", index=False)


if __name__ == "__main__":
    main()

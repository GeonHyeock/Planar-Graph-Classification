from typing import Any, Dict

import torch
import pandas as pd
import os
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning import LightningModule
from torchmetrics import MeanMetric
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from collections import defaultdict


class GCnetLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        DataVersion: str,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=True)

        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss()

        train_metric = {
            "train/Accuracy": Accuracy(task="multiclass", num_classes=net.classes),
            "train/Precision": Precision(task="multiclass", num_classes=net.classes),
            "train/Recall": Recall(task="multiclass", num_classes=net.classes),
            "train/F1Score": F1Score(task="multiclass", num_classes=net.classes),
        }

        valid_metric = {
            "valid/Accuracy": Accuracy(task="multiclass", num_classes=net.classes),
            "valid/Precision": Precision(task="multiclass", num_classes=net.classes),
            "valid/Recall": Recall(task="multiclass", num_classes=net.classes),
            "valid/F1Score": F1Score(task="multiclass", num_classes=net.classes),
        }

        test_metric = {
            "test/Accuracy": Accuracy(task="multiclass", num_classes=net.classes),
            "test/Precision": Precision(task="multiclass", num_classes=net.classes),
            "test/Recall": Recall(task="multiclass", num_classes=net.classes),
            "test/F1Score": F1Score(task="multiclass", num_classes=net.classes),
        }

        self.train_metric = torch.nn.ModuleDict(train_metric)
        self.valid_metric = torch.nn.ModuleDict(valid_metric)
        self.test_metric = torch.nn.ModuleDict(test_metric)

        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.test_result = defaultdict(list)

    def forward(self, x):
        return self.net(x)

    def on_train_start(self):
        pass

    def model_step(self, batch):
        logits = self.forward(batch["graph"])
        y = batch["colors"] - 1
        y = torch.where(y == 1, y, 0)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch):
        loss, preds, targets = self.model_step(batch)

        for name, metric in self.train_metric.items():
            metric.update(preds, targets)
        self.train_loss(loss)

        self.log_dict(self.train_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )

        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        for name, metric in self.valid_metric.items():
            metric.update(preds, targets)
        self.valid_loss(loss)

        self.log_dict(self.valid_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "valid/loss", self.valid_loss, on_step=False, on_epoch=True, prog_bar=True
        )

    def on_validation_epoch_end(self) -> None:
        pass

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)

        self.test_result["data_path"] += batch["data_path"]
        self.test_result["predict"] += preds.tolist()
        self.test_result["loss"] += [loss.tolist()]

        # update and log metrics
        for name, metric in self.test_metric.items():
            metric.update(preds, targets)
        self.test_loss(loss)

        self.log_dict(self.test_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )

    def on_test_epoch_end(self) -> None:
        result_df = pd.DataFrame(self.test_result)
        if isinstance(self.logger, WandbLogger):
            create_folder(f"../data/result/{self.hparams.DataVersion}")
            result_df.to_csv(
                f"../data/result/{self.hparams.DataVersion}/{self.logger._name}.csv",
                index=False,
            )
        else:
            result_df.to_csv(f"../data/result/my_result.csv", index=False)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.scheduler_monitor,
                    "interval": self.hparams.scheduler_interval,
                    "frequency": self.hparams.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


if __name__ == "__main__":
    _ = GCnetLitModule(None, None, None, None)

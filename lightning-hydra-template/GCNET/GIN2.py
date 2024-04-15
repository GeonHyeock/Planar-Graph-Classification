import torch
from GnnComparison.models.graph_classifiers.GIN import GIN
from GCNET.layer import Emb


class GIN2(GIN):
    def __init__(
        self,
        embedding_size,
        dim_target,
        hidden_units,
        train_eps,
        aggregation,
        dropout,
    ):
        super().__init__(10, embedding_size, dim_target, hidden_units, train_eps, aggregation, dropout)
        self.emb = Emb(embedding_size)

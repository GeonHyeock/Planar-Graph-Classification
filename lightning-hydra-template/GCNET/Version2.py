from GCNET.model import GCN, GCnet
from GnnComparison.models.graph_classifiers.GraphSAGE import GraphSAGE
from GnnComparison.models.graph_classifiers.GIN import GIN
from pytorchGAT.models.definitions.GAT import GAT
from pytorchGAT.utils.constants import LayerType
from GnnComparison.models.graph_classifiers.DGCNN import DGCNN
from GCNET.layer import Emb


class GCN2(GCN):
    def __init__(
        self, embedding_size, in_channels, hidden_channels, num_layers, out_channels, dropout, last_layer_dim, is_prob
    ):
        super().__init__(
            1, embedding_size, in_channels, hidden_channels, num_layers, out_channels, dropout, last_layer_dim, is_prob
        )

        self.emb = Emb(embedding_size)


class GraphSAGE2(GraphSAGE):
    def __init__(self, embedding_size, dim_features, dim_target, aggregation, num_layers):
        super().__init__(1, embedding_size, dim_features, dim_target, aggregation, num_layers)
        self.emb = Emb(embedding_size)


class GIN2(GIN):
    def __init__(self, embedding_size, dim_target, hidden_units, train_eps, aggregation, dropout):
        super().__init__(1, embedding_size, dim_target, hidden_units, train_eps, aggregation, dropout)
        self.emb = Emb(embedding_size)


class GAT2(GAT):
    def __init__(
        self,
        embedding_size,
        num_of_layers,
        num_heads_per_layer,
        num_features_per_layer,
        add_skip_connection=True,
        bias=True,
        dropout=0.6,
        layer_type=LayerType.IMP3,
        log_attention_weights=False,
        last_layer_dim=[64, 2],
        is_prob=True,
    ):
        super().__init__(
            1,
            embedding_size,
            num_of_layers,
            num_heads_per_layer,
            num_features_per_layer,
            add_skip_connection,
            bias,
            dropout,
            layer_type,
            log_attention_weights,
            last_layer_dim,
            is_prob,
        )
        self.emb = Emb(embedding_size)


class DGCNN2(DGCNN):
    def __init__(self, embedding_size, dim_features, num_layers, dense_dim, dim_target, k):
        super().__init__(1, embedding_size, dim_features, num_layers, dense_dim, dim_target, k)
        self.emb = Emb(embedding_size)


class GCnet2(GCnet):
    def __init__(self, embedding_size, n_layers, hidden_size, n_head, drop_prob, last_layer_dim, is_prob):
        super().__init__(1, embedding_size, n_layers, hidden_size, n_head, drop_prob, last_layer_dim, is_prob)
        self.emb = Emb(embedding_size)

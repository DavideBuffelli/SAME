from torch.nn import BatchNorm1d
import torch.nn.functional as F
from torchmeta.modules import MetaModule, MetaSequential, MetaLinear, MetaBatchNorm1d

from meta_graphconv import MetaGraphConv
from torch_geometric.nn import GCNConv
from task_output_layers import *


class MetaGCN(MetaModule):
    def __init__(self, in_dim, node_embedding_dim, residual_con=False, normalize_emb=False, batch_norm=False, dropout=False):
        super(MetaGCN, self).__init__()
        self.gcn_1 = MetaGraphConv(in_dim, node_embedding_dim)
        self.gcn_2 = MetaGraphConv(node_embedding_dim, node_embedding_dim)
        self.gcn_3 = MetaGraphConv(node_embedding_dim, node_embedding_dim)
        self.residual_con = residual_con
        self.normalize_emb = normalize_emb
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn1 = MetaBatchNorm1d(node_embedding_dim)
            self.bn2 = MetaBatchNorm1d(node_embedding_dim)
            self.bn3 = MetaBatchNorm1d(node_embedding_dim)
        self.dropout = dropout
        
    def forward(self, inputs, params=None):
        x = self.gcn_1(inputs.x, inputs.edge_index, params=get_subdict(params, 'gcn_1'))
        if self.batch_norm:
            x = self.bn1(x, params=get_subdict(params, 'bn1'))
        x = F.relu(x)
        #if self.dropout:
        #     x = F.dropout(x, training=self.training)
        if self.normalize_emb:
            x = F.normalize(x, p=2, dim=1)

        residual2 = x
        x = self.gcn_2(x, inputs.edge_index, params=get_subdict(params, 'gcn_2'))
        if self.batch_norm:
            x = self.bn2(x, params=get_subdict(params, 'bn2'))
        x = F.relu(x)
        if self.residual_con:
            x = x + residual2
        if self.normalize_emb:
            x = F.normalize(x, p=2, dim=1)

        residual3 = x
        x = self.gcn_3(x, inputs.edge_index, params=get_subdict(params, 'gcn_3'))
        if self.batch_norm:
            x = self.bn3(x, params=get_subdict(params, 'bn3'))
        x = F.relu(x)
        if self.residual_con:
            x = x + residual3
        if self.normalize_emb:
            x = F.normalize(x, p=2, dim=1)

        return x


class MetaOutputLayers(MetaModule):
    def __init__(self, node_embedding_dim, nc_num_classes, gc_num_classes):
        super(MetaOutputLayers, self).__init__()
        
        ### try nn.ModuleDict
        self.nc_output_layer = NodeClassificationOutputModule(node_embedding_dim, nc_num_classes)
        self.gc_output_layer = GraphClassificationOutputModule(node_embedding_dim, node_embedding_dim, gc_num_classes)
        self.lp_output_layer = LinkPredictionOutputModule(node_embedding_dim)
    
    def forward(self, node_embs, inputs, task_selector, params):
        if task_selector == "nc":
            x = self.nc_output_layer(node_embs, params=get_subdict(params, 'nc_output_layer'))
        elif task_selector == "gc":
            x = self.gc_output_layer(node_embs, 
                                     inputs.batch, 
                                     params=get_subdict(params, 'gc_output_layer'))
        elif  task_selector == "lp":
            x = self.lp_output_layer(node_embs, 
                                     inputs.pos_edge_index, 
                                     inputs.neg_edge_index, 
                                     params=get_subdict(params, 'lp_output_layer'))
        else:
            print("Invalid task selector.")
        
        return x
        

class MultitaskGCN(MetaModule):
    """All parameters are adapted in inner loop, and all are updated in outer loop."""
    def __init__(self, in_dim, node_embedding_dim, nc_num_classes, gc_num_classes, residual_con=False, normalize_emb=False, batch_norm=False, dropout=False):
        super(MultitaskGCN, self).__init__()
        
        self.name = "Meta_MultitaskGCN_MAML"

        self.gcn = MetaGCN(in_dim, node_embedding_dim, residual_con, normalize_emb, batch_norm, dropout)
        self.output_layer = MetaOutputLayers(node_embedding_dim, nc_num_classes, gc_num_classes)
        self.dropout = dropout

    def forward(self, inputs, task_selector=None, params=None, return_embeddings=False):
        x = self.gcn(inputs, params=get_subdict(params, 'gcn'))
        
        if return_embeddings:
            return x
        if task_selector == None:
            print("You need to specify a task selector")
            exit()

        if self.dropout:
            x = F.dropout(x, training=self.training)

        if isinstance(task_selector, list): # we are in the concurrent case
            out = {}
            for t in task_selector:
                out[t] = self.output_layer(x, inputs, t, params=get_subdict(params, 'output_layer'))
            return out
        else:
            x = self.output_layer(x, inputs, task_selector, params=get_subdict(params, 'output_layer'))
            return x
        

class MultitaskGCN_2(MetaModule):
    """Only output layers are adapted in inner loop, and all parameters are updated in outer loop."""
    def __init__(self, in_dim, node_embedding_dim, nc_num_classes, gc_num_classes, residual_con=False, normalize_emb=False, batch_norm=False, dropout=False):
        super(MultitaskGCN_2, self).__init__()
        
        self.name = "Meta_MultitaskGCN_ANIL"

        self.gcn_1 = GCNConv(in_dim, node_embedding_dim)
        self.gcn_2 = GCNConv(node_embedding_dim, node_embedding_dim)
        self.gcn_3 = GCNConv(node_embedding_dim, node_embedding_dim)
        self.output_layer = MetaOutputLayers(node_embedding_dim, nc_num_classes, gc_num_classes)
        self.residual_con = residual_con
        self.normalize_emb = normalize_emb
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn1 = BatchNorm1d(node_embedding_dim)
            self.bn2 = BatchNorm1d(node_embedding_dim)
            self.bn3 = BatchNorm1d(node_embedding_dim)
        self.dropout = dropout

    def forward(self, inputs, task_selector=None, params=None, return_embeddings=False):
        x = self.gcn_1(inputs.x, inputs.edge_index)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        #if self.dropout:
        #     x = F.dropout(x, training=self.training)
        if self.normalize_emb:
            x = F.normalize(x, p=2, dim=1)

        residual2 = x
        x = self.gcn_2(x, inputs.edge_index)
        if self.batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        if self.residual_con:
            x = x + residual2
        if self.normalize_emb:
            x = F.normalize(x, p=2, dim=1)

        residual3 = x
        x = self.gcn_3(x, inputs.edge_index)
        if self.batch_norm:
            x = self.bn3(x)
        x = F.relu(x)
        if self.residual_con:
            x = x + residual3
        if self.normalize_emb:
            x = F.normalize(x, p=2, dim=1)

        if return_embeddings:
            return x
        if task_selector == None:
            print("You need to specify a task selector")
            exit()

        if self.dropout:
             x = F.dropout(x, training=self.training)

        #x = self.output_layer(x, inputs, task_selector, params=get_subdict(params, 'output_layer'))
        if isinstance(task_selector, list): # we are in the concurrent case
            out = {}
            for t in task_selector:
                out[t] = self.output_layer(x, inputs, t, params=get_subdict(params, 'output_layer'))
            return out
        else:
            x = self.output_layer(x, inputs, task_selector, params=get_subdict(params, 'output_layer'))
            return x

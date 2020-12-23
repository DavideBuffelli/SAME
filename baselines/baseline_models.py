import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool


class NodeClassificationOutputModule(nn.Module):
    def __init__(self, node_embedding_dim, num_classes):
        super(NodeClassificationOutputModule, self).__init__()
        self.linear = nn.Linear(node_embedding_dim, num_classes)
        
    def forward(self, inputs):
        x = self.linear(inputs)
        return x
    
    
class GraphClassificationOutputModule(nn.Module):
    def __init__(self, node_embedding_dim, hidden_dim, num_classes):
        super(GraphClassificationOutputModule, self).__init__()
        self.linear1 = nn.Linear(node_embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs, batch):
        x = self.linear1(inputs)
        x = F.relu(x)
        x = global_add_pool(x, batch)
        x = self.linear2(x)
        return x
    

class LinkPredictionOutputModule(nn.Module):
    def __init__(self, node_embedding_dim):
        super(LinkPredictionOutputModule, self).__init__()
        self.linear_a = nn.Linear(node_embedding_dim, node_embedding_dim)
        #self.linear_b = nn.Linear(node_embedding_dim, node_embedding_dim)
        self.linear = nn.Linear(2*node_embedding_dim, 1)
        
    def forward(self, inputs, pos_edge_index, neg_edge_index):
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        node_a = torch.index_select(inputs, 0, total_edge_index[0])
        node_a = self.linear_a(node_a)
        node_b = torch.index_select(inputs, 0, total_edge_index[1])
        node_b = self.linear_a(node_b)
        x = torch.cat((node_a, node_b), 1)
        x = self.linear(x)
        x = torch.clamp(torch.sigmoid(x), min=1e-8, max=1 - 1e-8)
        return x


class SingleTaskGCN(nn.Module):
    def __init__(self, task, in_dim, node_embedding_dim, num_outputs, residual_con=False, normalize_emb=False, batch_norm= False, dropout=False):
        super(SingleTaskGCN, self).__init__()
        
        self.name = "Baseline_SingletaskGCN"
        self.task = task
        
        self.gcn_1 = GCNConv(in_dim, node_embedding_dim)
        self.gcn_2 = GCNConv(node_embedding_dim, node_embedding_dim)
        self.gcn_3 = GCNConv(node_embedding_dim, node_embedding_dim)
        
        if task == "gc":
            self.output_layer = GraphClassificationOutputModule(node_embedding_dim, node_embedding_dim, num_outputs)
        elif task == "nc":
            self.output_layer = NodeClassificationOutputModule(node_embedding_dim, num_outputs)
        elif task == "lp":
            self.output_layer = LinkPredictionOutputModule(node_embedding_dim)
            
        self.residual_con = residual_con
        self.normalize_emb = normalize_emb
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(node_embedding_dim)
            self.bn2 = nn.BatchNorm1d(node_embedding_dim)
            self.bn3 = nn.BatchNorm1d(node_embedding_dim)
        self.dropout = dropout

    def forward(self, inputs, return_embeddings=False):
        x = self.gcn_1(inputs.x, inputs.edge_index)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
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

        if self.dropout:
             x = F.dropout(x, training=self.training)

        if self.task == "gc":
            x = self.output_layer(x, inputs.batch)
        elif self.task == "nc":
            x = self.output_layer(x)
        elif self.task == "lp":
            x = self.output_layer(x, inputs.pos_edge_index, inputs.neg_edge_index)
        
        return x


class MultiTaskGCN(nn.Module):
    def __init__(self, tasks, in_dim, node_embedding_dim, num_gc_outputs, num_nc_outputs, residual_con=False, normalize_emb=False, batch_norm=False, dropout=False):
        super(MultiTaskGCN, self).__init__()
        
        self.name = "Baseline_MultitaskGCN"

        self.gcn_1 = GCNConv(in_dim, node_embedding_dim)
        self.gcn_2 = GCNConv(node_embedding_dim, node_embedding_dim)
        self.gcn_3 = GCNConv(node_embedding_dim, node_embedding_dim)
        
        self.tasks = tasks
        #if "gc" in tasks:
        self.gc_output_layer = GraphClassificationOutputModule(node_embedding_dim, node_embedding_dim, num_gc_outputs)
        #if "nc" in tasks:
        self.nc_output_layer = NodeClassificationOutputModule(node_embedding_dim, num_nc_outputs)
        #if "lp" in tasks:
        self.lp_output_layer = LinkPredictionOutputModule(node_embedding_dim)
            
        self.residual_con = residual_con
        self.normalize_emb = normalize_emb
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(node_embedding_dim)
            self.bn2 = nn.BatchNorm1d(node_embedding_dim)
            self.bn3 = nn.BatchNorm1d(node_embedding_dim)
        self.dropout = dropout

    def forward(self, inputs, return_embeddings=False):
        x = self.gcn_1(inputs.x, inputs.edge_index)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
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

        if self.dropout:
             x = F.dropout(x, training=self.training)

        gc_output = self.gc_output_layer(x, inputs.batch) if "gc" in self.tasks else None
        nc_output = self.nc_output_layer(x) if "nc" in self.tasks else None
        lp_output = self.lp_output_layer(x, inputs.pos_edge_index, inputs.neg_edge_index) if "lp" in self.tasks else None
        
        return gc_output, nc_output, lp_output

import torch
import torch.nn.functional as F
from torchmeta.modules import MetaModule, MetaLinear
from torchmeta.modules.utils import get_subdict
from torch_geometric.nn import global_add_pool


class NodeClassificationOutputModule(MetaModule):
    def __init__(self, node_embedding_dim, num_classes):
        super(NodeClassificationOutputModule, self).__init__()
        self.linear = MetaLinear(node_embedding_dim, num_classes)
        
    def forward(self, inputs, params=None):
        x = self.linear(inputs, params=get_subdict(params, 'linear'))
        return x
    
    
class GraphClassificationOutputModule(MetaModule):
    def __init__(self, node_embedding_dim, hidden_dim, num_classes):
        super(GraphClassificationOutputModule, self).__init__()
        self.linear1 = MetaLinear(node_embedding_dim, hidden_dim)
        self.linear2 = MetaLinear(hidden_dim, num_classes)

    def forward(self, inputs, batch, params=None):
        x = self.linear1(inputs, params=get_subdict(params, 'linear1'))
        x = F.relu(x)
        x = global_add_pool(x, batch)
        x = self.linear2(x, params=get_subdict(params, 'linear2'))
        return x
    
    
class LinkPredictionOutputModule(MetaModule):
    def __init__(self, node_embedding_dim):
        super(LinkPredictionOutputModule, self).__init__()
        self.linear_a = MetaLinear(node_embedding_dim, node_embedding_dim)
        self.linear = MetaLinear(2*node_embedding_dim, 1)
        
    def forward(self, inputs, pos_edge_index, neg_edge_index, params=None):
        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        node_a = torch.index_select(inputs, 0, total_edge_index[0])
        node_a = self.linear_a(node_a, params=get_subdict(params, 'linear_a'))
        node_b = torch.index_select(inputs, 0, total_edge_index[1])
        node_b = self.linear_a(node_b, params=get_subdict(params, 'linear_a'))
        x = torch.cat((node_a, node_b), 1)
        x = self.linear(x, params=get_subdict(params, 'linear'))
        x = torch.clamp(torch.sigmoid(x), min=1e-8, max=1 - 1e-8)
        return x

import torch
from collections import OrderedDict
from torch_geometric.nn.conv import GCNConv
from torchmeta.modules import MetaModule


class MetaGraphConv(GCNConv, MetaModule):
    """Adaptation of GCNConv from PyTorch Geometric which inherits from MetaModule and
    introduces the 'params' parameter in the forward method. This way the network can use
    external weights to perform the 'forward' operations, and it becomes much easier to 
    perform the inner loop optimization step during meta-learning. For other MetaModules
    refer to torchmeta.modules
    """
    def update(self, aggr_out):
        # In GCNConv the bias is applied here (if used). We move it to the end of 'forward'
        # so that we can use external weights
        return aggr_out
        
    def forward(self, x, edge_index, edge_weight=None, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        
        #x = torch.matmul(x, self.weight)
        x = torch.matmul(x, params['weight'])
    
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    "Cached {} number of edges, but found {}. Please "
                    "disable the caching behavior of this layer by removing "
                    "the `cached=True` argument in its constructor.".format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(
                    self.node_dim), edge_weight, self.improved, x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        x = self.propagate(edge_index, x=x, norm=norm)
        
        if self.bias is not None:
            #x = x + self.bias
            x = x + bias
        
        return x
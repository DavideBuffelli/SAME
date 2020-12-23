import math
import numpy as np
import random
import torch
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected, add_remaining_self_loops


def print_dataset_stats(dataset):
    print("--- {} Dataset Statistichs:".format(dataset.name))
    print("Number of Graphs: {}".format(len(dataset)))
    print("Number of Graph Classes: {}".format(dataset.num_classes))
    print("Number of Node Features: {}".format(dataset.num_node_features))
    print("Number of Node Classes: {}".format(dataset[0].node_y.size(1)))
    print("Number of Edge Features: {} \n".format(dataset.num_edge_features))


class DataSizeFilter(object):
    def __init__(self, min_num_nodes, max_num_nodes):
            self.min_num_nodes = min_num_nodes if min_num_nodes else 0
            self.max_num_nodes = max_num_nodes if max_num_nodes else math.inf
            
    def __call__(self, data):
        return data.num_nodes >= self.min_num_nodes and data.num_nodes <= self.max_num_nodes

        
class SeparateNodeFeaturesAndLabels(object):   
    def __init__(self, num_node_features):
            self.num_node_features = num_node_features
               
    def __call__(self, data):
        node_features = data.x[:, :self.num_node_features]
        node_labels = data.x[:, self.num_node_features:]
        data.x = node_features
        data.node_y = node_labels
        return data


def get_graph_dataset(name, destination_dir="", min_num_nodes=None, max_num_nodes=None):
    """ Get a dataset from the TUD library (https://chrsmrrs.github.io/datasets/docs/home/) """
    if destination_dir == None:
        destination_dir = ""

    if name == "ENZYMES":
        num_node_features = 18
        train_val_test_ratio = [0.7, 0.1, 0.2]
    elif name == "PROTEINS":
        name = "PROTEINS_full"
        num_node_features = 29
        train_val_test_ratio = [0.7, 0.1, 0.2]
    elif name == "DHFR":
        num_node_features = 3
        train_val_test_ratio = [0.7, 0.1, 0.2]
    elif name == "COX2":
        num_node_features = 3
        train_val_test_ratio = [0.7, 0.2, 0.1]

    dataset = TUDataset(root=destination_dir, 
                        name=name, 
                        use_node_attr=True,
                        pre_filter=DataSizeFilter(min_num_nodes, max_num_nodes),
                        pre_transform=SeparateNodeFeaturesAndLabels(num_node_features), 
                        transform=NormalizeFeatures())
    print_dataset_stats(dataset)
    return dataset, train_val_test_ratio
    

def get_dataloaders(dataset, batch_size, batch_task, train_val_test_ratio=[0.7, 0.1, 0.2], num_workers=1, shuffle_train=False):
    num_training_graphs = math.floor(len(dataset)*train_val_test_ratio[0])
    num_validation_graphs = math.floor(len(dataset)*train_val_test_ratio[1])
    meta_train_dataset = dataset[:num_training_graphs]
    meta_val_dataset = dataset[num_training_graphs:num_training_graphs+num_validation_graphs]
    meta_test_dataset = dataset[num_training_graphs+num_validation_graphs:]

    min_batch_size = 1
    if batch_task == "conc":
        min_batch_size = 4
    elif batch_task == "multi":
        min_batch_size = 6
    meta_train_dataloader = DataLoader(meta_train_dataset, 
                                       batch_size=batch_size, 
                                       shuffle=shuffle_train,
                                       drop_last=(len(meta_train_dataset)%batch_size < min_batch_size), 
                                       num_workers=num_workers) 
    meta_val_dataloader = None
    if len(meta_val_dataset) > 0:
        meta_val_dataloader = DataLoader(meta_val_dataset, 
                                          batch_size=6,
                                          shuffle=False,
                                          drop_last=(len(meta_val_dataset)%6 < min_batch_size), 
                                          num_workers=num_workers) 
    meta_test_dataloader = None
    if len(meta_test_dataset) > 0:
        meta_test_dataloader = DataLoader(meta_test_dataset, 
                                          batch_size=6,
                                          shuffle=False,
                                          drop_last=(len(meta_test_dataset)%6 < min_batch_size), 
                                          num_workers=num_workers) 
    return meta_train_dataloader, meta_val_dataloader, meta_test_dataloader
                                      

def remove_double_edges(edge_index):
    """In undirected graphs you have both (i, j) and (j, i) for every edge. 
    This method removes (j, i)."""
    row, col = edge_index
    mask = row <= col
    row, col = row[mask], col[mask]
    ei_without_double_edges = torch.stack([row, col], dim=0).long()
    return ei_without_double_edges


def negative_sampling(edge_index, num_nodes, num_neg_samples=None, shuffle_neg_egdes=True):
    """Return an edge index containing edges that are not present in the graph spanned by
    the input edge_index."""

    # edge_index must already contain self-loops
    num_neg_samples = num_neg_samples or edge_index.size(1)

    # Handle '|V|^2 - |E| - |V| < |E|' case for G = (V, E).
    # (We only want edges that are not in the graph, we don't want directed duplicate edges
    # (if we take (i,j) we don't want (j,i)), and that are not self-loops)
    num_neg_samples = min(num_neg_samples,
                          num_nodes * num_nodes - edge_index.size(1) - num_nodes)

    # Upper triangle indices: N + ... + 1 = N (N + 1) / 2
    rng = range((num_nodes * (num_nodes + 1)) // 2)
    # Remove edges in the lower triangle matrix.
    row, col = edge_index
    mask = row <= col
    row, col = row[mask], col[mask]
    # Assign an index to each position in the upper triangular matrix
    # idx = N * i + j - i * (i+1) / 2
    pos_edges_idx = (row * num_nodes + col - row * (row + 1) // 2).to('cpu')

    # Get all 'available' indexes, and out of these randomly take the ones for negative examples 
    all_possible_indexes = torch.tensor(rng)
    mask = torch.from_numpy(np.isin(all_possible_indexes, pos_edges_idx)).to(torch.bool)
    remaining = all_possible_indexes[mask == False]
    if shuffle_neg_egdes:
        perm = torch.randperm(remaining.size(0))
        neg_samples = remaining[perm][:num_neg_samples]
    else:
        neg_samples = remaining[:num_neg_samples]

    # Go back from indexes to (row, col)
    # (-sqrt((2 * N + 1)^2 - 8 * perm) + 2 * N + 1) / 2
    row = torch.floor((-torch.sqrt((2. * num_nodes + 1.)**2 - 8. * neg_samples) +
                       2 * num_nodes + 1) / 2)
    col = neg_samples - row * (2 * num_nodes - row - 1) // 2
    neg_edge_index = torch.stack([row, col], dim=0).long()

    return neg_edge_index.to(edge_index.device)


def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1)).float()
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels.to(pos_edge_index.device)
    

def prepare_data_for_link_prediction(datalist, train_ratio=0.8, neg_to_pos_edge_ratio=1, rnd_labeled_edges=True):
    """For each graph it splits the edges in training and testing (both with also a
    negative set of examples).
    rnd_labeled_edges=True means that the positive and negative edges for training are choosen at random (at 
    different epochs, the same graph can have different positive/negative edges chosen for training)."""
    train_data_list = []
    test_data_list = []
    for graph in datalist:
        train_graph = graph
        test_graph = train_graph.clone() 
            
        # Create Negative edges examples
        ei_without_double_edges = remove_double_edges(graph.edge_index)
        ei_with_self_loops, _ = add_remaining_self_loops(ei_without_double_edges, 
                                                         num_nodes=graph.num_nodes)

        neg_edge_index = negative_sampling(edge_index=ei_with_self_loops, 
                                           num_nodes=graph.num_nodes,
                                           num_neg_samples=neg_to_pos_edge_ratio*ei_without_double_edges.size(1),
                                           shuffle_neg_egdes=rnd_labeled_edges)

        num_train_pos_edges = math.floor(ei_without_double_edges.size(1)*train_ratio)
        num_train_neg_edges = math.floor(neg_edge_index.size(1)*train_ratio)
        
        # Split Positive edges
        if rnd_labeled_edges:
            perm = torch.randperm(ei_without_double_edges.size(1))
            row, col = ei_without_double_edges[0][perm], ei_without_double_edges[1][perm]
        else:
            row, col = ei_without_double_edges[0], ei_without_double_edges[1]
        train_graph.pos_edge_index = torch.stack([row[:num_train_pos_edges], col[:num_train_pos_edges]], dim=0)
        test_graph.pos_edge_index = torch.stack([row[num_train_pos_edges:], col[num_train_pos_edges:]], dim=0)
        
        # Update edge_index for message-passing for link prediction (no test edges)
        train_graph.edge_index = to_undirected(train_graph.pos_edge_index, num_nodes=train_graph.num_nodes)
        test_graph.edge_index = train_graph.edge_index
        
        # Split Negative edges
        if rnd_labeled_edges:
            perm = torch.randperm(neg_edge_index.size(1))
            row, col = neg_edge_index[0][perm], neg_edge_index[1][perm]
        else:
            row, col = neg_edge_index[0], neg_edge_index[1]
        train_graph.neg_edge_index = torch.stack([row[:num_train_neg_edges], col[:num_train_neg_edges]], dim=0)
        test_graph.neg_edge_index = torch.stack([row[num_train_neg_edges:], col[num_train_neg_edges:]], dim=0)
        
        train_data_list.append(train_graph)
        test_data_list.append(test_graph)
    
    return  train_data_list, test_data_list


def prepare_data_for_node_classification(datalist, train_ratio=0.3, rnd_labeled_nodes=True):
    """For each graph split the nodes for training and testing. It creates a train_mask
    ehere elements equal to 1 are for training.
    rnd_labeled_nodes=True means that the nodes that are given labels for training are choosen at random (at 
    different epochs, the same graph can have different labeled nodes)."""
    for graph in datalist:
        num_nodes = graph.num_nodes
        num_classes = graph.node_y.size(1)
        nodes_per_class = {}
        for node, y in enumerate(graph.node_y.argmax(1)):
            y = y.item()
            if y not in nodes_per_class:
                nodes_per_class[y] = []
            nodes_per_class[y].append(node)
        
        train_mask = torch.zeros((num_nodes, 1))
        for y in nodes_per_class.keys():
            num_nodes_in_class = len(nodes_per_class[y])
            num_train_nodes = math.floor(num_nodes_in_class*train_ratio)
            if rnd_labeled_nodes:
                train_nodes = random.sample(nodes_per_class[y], num_train_nodes)
            else:
                train_nodes = nodes_per_class[y][:num_train_nodes]
            for node in train_nodes:
                train_mask[node] = 1
        graph.train_mask = train_mask
    return datalist
    

def prepare_data_for_graph_classification(datalist, train_ratio=0.6, rnd_labeled_graphs=True):
    """Split the graphs in training and testing based in train_ratio.
    rnd_labeled_graphs=True means that the graphs in the batch that are used for training are choosen at random
    (at each epoch, for the same batch, different graphs are used for training)."""
    num_train_examples = math.floor(len(datalist)*train_ratio)
    if rnd_labeled_graphs:
        random.shuffle(datalist)
    train_list = datalist[:num_train_examples]
    test_list = datalist[num_train_examples:]
    return train_list, test_list

   
def create_batch_task_list(num_batches, tasks=["gc", "nc", "lp"]):
    num_batches_per_task = math.floor(num_batches/len(tasks))
    batch_task_list = []
    for i in range(num_batches):
        ith_task_idx = math.floor(i/num_batches_per_task)
        if ith_task_idx >= len(tasks):
            ith_task_idx = i % len(tasks)
        batch_task_list.append(tasks[ith_task_idx])
    random.shuffle(batch_task_list)
    return batch_task_list


""" DATA FORMAT
A Batch of Tasks for Meta-Training or Meta-Testing is provided in a tuple (A, B, C) where:
- A is a list of strings
- B is a list of training batches
- C is a list of testing batches

e.g.:
A[i] is the string (or list of strings) representing the task(s) of the i-th batch (e.g. "gc", "np", "lp", ["gc","nc"])
B[i] is the batch containing the examples for Meta-Train/Test training the model (adapt 
    phase) for i-th task(s)
C[i] is the batch containing the examples for Meta-Train/Test testing the model for i-th task(s)
"""
def single_task_train_test_split(batch, rnd_batch_labels, task=None):
    """Each batch contains examples of only one task."""
    if task:
        task_for_current_batch = task
    else:
        tasks = ["gc", "nc", "lp"]
        task_for_current_batch = random.choice(tasks)

    if task_for_current_batch == "gc":
        train_ratio = 0.6
        list_batch = batch.to_data_list()
        train_list, test_list = prepare_data_for_graph_classification(list_batch, train_ratio, rnd_batch_labels)
        train_batch = Batch.from_data_list(train_list)
        test_batch = Batch.from_data_list(test_list)
        return ["gc"], [train_batch], [test_batch]
    elif task_for_current_batch == "nc":
        train_ratio = 0.3
        list_batch = batch.to_data_list()
        list_batch = prepare_data_for_node_classification(list_batch, train_ratio, rnd_batch_labels)
        train_batch = Batch.from_data_list(list_batch)
        return ["nc"], [train_batch], [None]
    elif task_for_current_batch == "lp":
        train_ratio = 0.8
        neg_to_pos_edge_ratio = 1
        list_batch = batch.to_data_list()
        train_data_list, test_data_list = prepare_data_for_link_prediction(list_batch, 
                                                                           train_ratio,
                                                                           neg_to_pos_edge_ratio,
                                                                           rnd_batch_labels)
        train_batch = Batch.from_data_list(train_data_list)
        test_batch = Batch.from_data_list(test_data_list)
        return ["lp"], [train_batch], [test_batch] 
        
        
def multi_task_train_test_split(batch, rnd_batch_labels, tasks=["gc", "nc", "lp"]):
    """Each batch contains examples of all task (but for each graph in the batch only one 
    task is performed on it)."""
    if len(tasks) < 2:
        print("Can't use multi with only one task. single_task_train_test_split will be used instead.")
        return single_task_train_test_split(batch, rnd_batch_labels)
    if batch.num_graphs < len(tasks)*2:
        print("Batch size too small for multi_task_train_test_split. You need at least 2 graphs per task.")
        print("single_task_train_test_split will be used instead.")
        return single_task_train_test_split(batch, rnd_batch_labels)
    
    num_graphs_per_task = math.floor(batch.num_graphs/len(tasks))
    list_batch = batch.to_data_list()
    graphs_per_task = {task: [] for task in tasks}
    for i, graph in enumerate(list_batch):
        ith_task_idx = math.floor(i/num_graphs_per_task)
        if ith_task_idx >= len(tasks):
            ith_task_idx = i % len(tasks)
        graphs_per_task[tasks[ith_task_idx]].append(list_batch[i])

    task_strings = []; train_batch = []; test_batch = []
    if "gc" in tasks:
        gc_train_ratio = 0.6
        gc_train_list, gc_test_list = prepare_data_for_graph_classification(graphs_per_task["gc"], gc_train_ratio, rnd_batch_labels)
        gc_train_batch = Batch.from_data_list(gc_train_list)
        gc_test_batch = Batch.from_data_list(gc_test_list)
        task_strings.append("gc")
        train_batch.append(gc_train_batch)
        test_batch.append(gc_test_batch)
    
    if "nc" in tasks:
        nc_train_ratio = 0.3
        nc_train_list = prepare_data_for_node_classification(graphs_per_task["nc"], nc_train_ratio, rnd_batch_labels)
        nc_train_batch = Batch.from_data_list(nc_train_list)
        task_strings.append("nc")
        train_batch.append(nc_train_batch)
        test_batch.append(None)
    
    if "lp" in tasks:
        lp_train_ratio = 0.8
        neg_to_pos_edge_ratio = 1
        lp_train_list, lp_test_list = prepare_data_for_link_prediction(graphs_per_task["lp"],
                                                                       lp_train_ratio,
                                                                       neg_to_pos_edge_ratio, 
                                                                       rnd_batch_labels)
        lp_train_batch = Batch.from_data_list(lp_train_list)
        lp_test_batch = Batch.from_data_list(lp_test_list)
        task_strings.append("lp")
        train_batch.append(lp_train_batch)
        test_batch.append(lp_test_batch)
    
    task_batch = (task_strings, train_batch, test_batch)
    return task_batch


def prepare_data_for_concurrent_multitask(list_batch, train, rnd_labels, tasks=["gc", "nc", "lp"]):
    # Graphs are already ready for Graph Classification 
    if rnd_labels:
        random.shuffle(list_batch)

    # For Node Classification, add a train mask to each node
    if "nc" in tasks:
        if train:
            nc_train_ratio = 1
        else:
            nc_train_ratio = 0
        list_batch = prepare_data_for_node_classification(list_batch, nc_train_ratio, rnd_labeled_nodes=rnd_labels)
    
    # For Link Prediction create positive and negative examples (we hide 1-lp_train_ratio % of the edges)
    if "lp" in tasks:
        lp_train_ratio = 0.8
        train_data_list, test_data_list = prepare_data_for_link_prediction(list_batch,
                                                                           train_ratio=lp_train_ratio,
                                                                           neg_to_pos_edge_ratio=1,
                                                                           rnd_labeled_edges=rnd_labels)
        if train:
            for train_data, test_data in zip(train_data_list, test_data_list):
                train_data.pos_edge_index = torch.cat((train_data.pos_edge_index, test_data.pos_edge_index), dim=1)
                train_data.neg_edge_index = torch.cat((train_data.neg_edge_index, test_data.neg_edge_index), dim=1)
            list_batch = train_data_list
        else:
            list_batch = test_data_list
    return list_batch


def concurrent_multi_task_train_test_split(batch, rnd_labels, tasks=["gc", "nc", "lp"]):
    """For every graph in the batch, all the tasks are performed concurrently."""
    if batch.num_graphs < 4:
        print("Error! Batch size too small for concurrent_multi_task_train_test_split. You need at least 4 graphs.")
        exit()
    
    list_batch = batch.to_data_list()
    num_train_graphs = math.ceil(len(list_batch)/2)
    train_list_batch = prepare_data_for_concurrent_multitask(list_batch[:num_train_graphs], True, rnd_labels, tasks) 
    test_list_batch = prepare_data_for_concurrent_multitask(list_batch[num_train_graphs:], False, rnd_labels, tasks) 
    train_batch = Batch.from_data_list(train_list_batch)
    test_batch = Batch.from_data_list(test_list_batch)
    return [tasks], [train_batch], [test_batch]

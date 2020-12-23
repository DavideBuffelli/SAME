import numpy as np
from tqdm import tqdm


class EpochStats:
    def __init__(self):
        self.tasks = ["gc", "nc", "lp"]
        self.task_correct = {task: [] for task in self.tasks}
        self.task_losses = {task: [] for task in self.tasks}
        self.num_gc_graphs = 0
        self.num_nc_nodes = 0
        self.num_lp_batches = 0
        self.num_lp_edges = 0
        
    def update(self, task, batch, loss, acc, train):
        if task == "gc":
            self.num_gc_graphs += batch.num_graphs
            self.task_losses[task].append((loss*batch.num_graphs).item())
            self.task_correct[task].append((acc*batch.num_graphs).item())
        elif task == "nc":
            num_train_nodes = batch.train_mask.sum().item()
            num_test_nodes = batch.batch.size(0) - num_train_nodes
            if train:
                self.num_nc_nodes += num_train_nodes
                self.task_losses[task].append((loss*num_train_nodes).item())
                self.task_correct[task].append((acc*num_train_nodes).item())
            else:
                self.num_nc_nodes += num_test_nodes
                self.task_losses[task].append((loss*num_test_nodes).item())
                self.task_correct[task].append((acc*num_test_nodes).item())
        elif task == "lp":
            self.num_lp_batches += 1
            num_lp_edges = batch.neg_edge_index.size(1) + batch.pos_edge_index.size(1)
            self.num_lp_edges += num_lp_edges 
            self.task_losses[task].append((loss*num_lp_edges).item())
            self.task_correct[task].append(acc.item())

    def get_average_stats(self):
        stats = {}
        for task in self.tasks:
            if len(self.task_correct[task]) == 0:
                continue
            
            stats[task] = {}
            if task == "gc":
                stats[task]['acc'] = np.stack(self.task_correct[task]).sum() / self.num_gc_graphs
                stats[task]['loss'] = np.stack(self.task_losses[task]).sum() / self.num_gc_graphs
            elif task == "nc":
                stats[task]['acc'] = np.stack(self.task_correct[task]).sum() / self.num_nc_nodes
                stats[task]['loss'] = np.stack(self.task_losses[task]).sum() / self.num_nc_nodes 
            elif task == "lp":
                stats[task]['acc'] = np.stack(self.task_correct[task]).sum() / self.num_lp_batches
                stats[task]['loss'] = np.stack(self.task_losses[task]).sum() / self.num_lp_edges
            
        return stats
 

def print_task_accs_and_losses(tasks_epoch_stats):
    str_stats = ""
    for task in tasks_epoch_stats:
        task_acc = tasks_epoch_stats[task]['acc']
        task_loss = tasks_epoch_stats[task]['loss']
        str_stats += f"{task:>4}:{task_loss:^10.4f}|{task_acc:^12.4f}\n"
    tqdm.write(str_stats)
    

def print_train_epoch_stats(epoch, tasks_epoch_stats):
    tqdm.write("Epoch: {}\nTask:   Loss   |  Accuracy  \n".format(epoch))
    print_task_accs_and_losses(tasks_epoch_stats)
    

def print_test_stats(tasks_epoch_stats):
    tqdm.write("\n\n--- Results on Test Data:\nTask:   Loss   |  Accuracy \n")
    print_task_accs_and_losses(tasks_epoch_stats)
    tqdm.write("\n\n")

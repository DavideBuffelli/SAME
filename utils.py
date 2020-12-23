import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import torch
from tqdm import tqdm
import random


def print_arguments(args):
    print("--- Arguments: ")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("---------------\n")

    
def set_seeds(seed=None):
    if not seed:
        seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
def get_accuracy(logits, targets):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())
    
    
def get_correct(logits, targets):
    _, predictions = torch.max(logits, dim=-1)
    return predictions.eq(targets).sum().float()


class EpochStats:
    def __init__(self):
        self.tasks = ["gc", "nc", "lp"]
        self.task_correct = {task: [] for task in self.tasks}
        self.task_inner_losses = {task: [] for task in self.tasks}
        self.task_outer_losses = {task: [] for task in self.tasks}
        self.num_gc_train_graphs = 0
        self.num_gc_test_graphs = 0
        self.num_nc_train_nodes = 0
        self.num_nc_test_nodes = 0
        self.num_lp_batches = 0
        self.nun_lp_train_edges = 0
        self.nun_lp_test_edges = 0
        
    def update(self, task, train_batch, test_batch, inner_loss, outer_loss, test_acc, conc=False):
        if task == "gc":
            num_train_graphs = train_batch.num_graphs 
            num_test_graphs = test_batch.num_graphs
            self.num_gc_train_graphs += num_train_graphs
            self.num_gc_test_graphs += num_test_graphs
            self.task_inner_losses[task].append((inner_loss*num_train_graphs).item())
            self.task_outer_losses[task].append((outer_loss*num_test_graphs).item())
            self.task_correct[task].append((test_acc*num_test_graphs).item())
        elif task == "nc":
            num_train_nodes = train_batch.train_mask.sum().item()
            num_test_nodes = train_batch.batch.size(0) - num_train_nodes
            if conc:
                num_test_nodes = test_batch.batch.size(0)
            self.num_nc_train_nodes += num_train_nodes
            self.num_nc_test_nodes += num_test_nodes
            self.task_inner_losses[task].append((inner_loss*num_train_nodes).item())
            self.task_outer_losses[task].append((outer_loss*num_test_nodes).item())
            self.task_correct[task].append((test_acc*num_test_nodes).item())
        elif task == "lp":
            num_train_edges = train_batch.neg_edge_index.size(1) + train_batch.pos_edge_index.size(1)
            num_test_edges = test_batch.neg_edge_index.size(1) + test_batch.pos_edge_index.size(1)
            self.num_lp_batches += 1
            self.nun_lp_train_edges += num_train_edges
            self.nun_lp_test_edges += num_test_edges
            self.task_inner_losses[task].append((inner_loss*num_train_edges).item())
            self.task_outer_losses[task].append((outer_loss*num_test_edges).item())
            self.task_correct[task].append(test_acc.item())

    def get_average_stats(self):
        stats = {}
        for task in self.tasks:
            if len(self.task_correct[task]) == 0:
                continue
            
            stats[task] = {}
            if task == "gc":
                train_divider = self.num_gc_train_graphs
                test_divider = self.num_gc_test_graphs
                stats[task]['acc'] = np.stack(self.task_correct[task]).sum() / test_divider
                stats[task]['inner_loss'] = np.stack(self.task_inner_losses[task]).sum() / train_divider
                stats[task]['outer_loss'] = np.stack(self.task_outer_losses[task]).sum() / test_divider
            elif task == "nc":
                train_divider = self.num_nc_train_nodes
                test_divider = self.num_nc_test_nodes
                stats[task]['acc'] = np.stack(self.task_correct[task]).sum() / test_divider
                stats[task]['inner_loss'] = np.stack(self.task_inner_losses[task]).sum() / train_divider
                stats[task]['outer_loss'] = np.stack(self.task_outer_losses[task]).sum() / test_divider
            elif task == "lp":
                train_divider = self.nun_lp_train_edges
                test_divider = self.nun_lp_test_edges
                stats[task]['acc'] = np.stack(self.task_correct[task]).sum() / self.num_lp_batches
                stats[task]['inner_loss'] = np.stack(self.task_inner_losses[task]).sum() / train_divider
                stats[task]['outer_loss'] = np.stack(self.task_outer_losses[task]).sum() / test_divider
            
        return stats
    

class BaselinesCVStats:
    def __init__(self):
        self.b_linear_svm_stats = {"gc": [], "nc": [], "lp": []}
        self.b_output_layer_test_stats = {"gc": [], "nc": [], "lp": []}
        self.b_trained_ol_test_stats = {"gc": [], "nc": [], "lp": []}
        self.b_finetuned_ol_test_stats = {"gc": [], "nc": [], "lp": []}

    def update(self, embedding_stats):
        for task in ["gc", "nc", "lp"]:
            self.b_linear_svm_stats[task].append(embedding_stats[task][0])
            self.b_output_layer_test_stats[task].append(embedding_stats[task][1])
            if len(embedding_stats[task]) == 4:
                self.b_trained_ol_test_stats[task].append(embedding_stats[task][2])
                self.b_finetuned_ol_test_stats[task].append(embedding_stats[task][3])

    def print_stats(self):
        print("\n\n############ Baselines on Embeddings ############")
        for task in ["gc", "nc", "lp"]:
            print(f"################ Task: {task} ################")
            print("##### Linear SVM")
            print_cv_stats_linear_svm(self.b_linear_svm_stats[task], task) 
            print("##### Trained from scratch")
            print_cv_stats(self.b_output_layer_test_stats[task]) 
            if len(self.b_trained_ol_test_stats[task]) > 0:
                print("##### Trained Mutitask")
                print_cv_stats(self.b_trained_ol_test_stats[task]) 
                print("##### Finetuned Mutitask")
                print_cv_stats(self.b_finetuned_ol_test_stats[task]) 


def save_model(model, output_folder, name, args=None):
    """ Saves the model state_dict, and if provided the arguments passed to train.py in order to reproduce the 
    hyperparameters. """
    # Move to CPU for better compatibility
    device = next(model.parameters()).device
    model = model.to("cpu")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    model_filename = os.path.join(output_folder, 'model_{}.pt'.format(name))
    torch.save(model.state_dict(), model_filename)
    
    if args:
        args_filename = os.path.join(output_folder, 'model_{}_args.txt'.format(name))
        with open(args_filename, 'w') as f:
            for arg in vars(args):
                arg_str = "{}: {}\n".format(arg, getattr(args, arg))
                f.write(arg_str)

    model.to(device)

    return model_filename
        

def recover_early_stopping_best_weights(model, early_stopping_tmp_dir, name="best_val", delete_dir=True):
    """ Sets the models wights to the weights saved in early_stopping_tmp_dir (the best weights found during training 
    while testing on validation set)."""
    best_model_path = os.path.join(early_stopping_tmp_dir, f"model_{name}.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model from early stopping.")
        if delete_dir:
            shutil.rmtree(early_stopping_tmp_dir, ignore_errors=True)
    else:
        print("No model saved for early stopping, maybe number of epochs was too small.")


def create_stats_plots(global_stats, cv_iteration=""):
    """global_stats is a list where each element corresponds to the stats of 1 epoch. The 
    stats for one epoch are a nested dict like the one returned by 'get_average_stats()'."""
    tasks = global_stats[0].keys()
    inner_losses_over_epochs = {task: [] for task in tasks}
    outer_losses_over_epochs = {task: [] for task in tasks}
    accs_over_epochs = {task: [] for task in tasks}
    for epoch_stats in global_stats:
        for task in epoch_stats:
            inner_losses_over_epochs[task].append(epoch_stats[task]['inner_loss'])
            outer_losses_over_epochs[task].append(epoch_stats[task]['outer_loss'])
            accs_over_epochs[task].append(epoch_stats[task]['acc'])
            
    if not os.path.exists("fig"):
        os.mkdir("fig")
     
    for title, data in zip(["Inner Loss", "Outer Loss", "Accuracy"], 
                           [inner_losses_over_epochs, outer_losses_over_epochs, accs_over_epochs]):
        fig, ax = plt.subplots()    
        
        for task in data:
            if task == "gc":
                color = "r"
                style = "--"
            elif task == "nc":
                color = "g"
                style = "dashdot"
            else:
                color = "b"
                style = "dotted"
            x_values = np.arange(len(data[task]))
            y_values = data[task]
            ax.plot(x_values, y_values, color=color, linestyle=style, label=task)

        ax.set(xlabel='Epoch', ylabel=title)
        ax.legend(loc='best')
        filename = "cv_"+str(cv_iteration)+"_"+title.replace(" ", "_")+".pdf"
        fig_path = os.path.join("fig", filename)
        fig.savefig(fig_path, format="pdf", bbox_inches = "tight")
    
  
def print_task_accs_and_losses(tasks_epoch_stats):
    str_stats = ""
    for task in tasks_epoch_stats:
        task_acc = tasks_epoch_stats[task]['acc']
        task_inner_loss = tasks_epoch_stats[task]['inner_loss']
        task_outer_loss = tasks_epoch_stats[task]['outer_loss']
        str_stats += f"{task:>4}:{task_inner_loss:^10.4f}|{task_outer_loss:^11.4f}|{task_acc:^10.4f}\n"
    tqdm.write(str_stats)
    

def print_train_epoch_stats(epoch, tasks_epoch_stats):
    tqdm.write("Epoch: {}\nTask: In. Loss | Out. Loss | Accuracy \n".format(epoch), end="")
    print_task_accs_and_losses(tasks_epoch_stats)
    

def print_test_stats(tasks_epoch_stats):
    tqdm.write("\n\n--- Results on Test Data:\nTask: In. Loss | Out. Loss | Accuracy \n", end="")
    print_task_accs_and_losses(tasks_epoch_stats)
    tqdm.write("\n\n")


def get_cv_task_accs(cv_stats):
    tasks = cv_stats[0].keys()
    accs = {task: [] for task in tasks}
    for cv_fold_stats in cv_stats:
        for task in cv_fold_stats:
            accuracy = cv_fold_stats[task]['acc']
            accs[task].append(accuracy)
    return accs
    

def print_cv_stats(cv_stats):
    accs = get_cv_task_accs(cv_stats)
    print("--- Cross Validation Results:")                
    print("-- Accuracies")
    str_stats = "Task:  Avg. +- Std.  |  Max.  |  Min.\n"
    for task in accs:
        accs_array = np.array(accs[task])
        avg = accs_array.mean()
        std = accs_array.std()
        min = accs_array.min()
        max = accs_array.max()
        str_stats += f"{task:>4}:{avg:>7.4f}+-{std:<7.4f}|{max:^8.4f}|{min:^8.4f}\n"
    print(str_stats)


def print_cv_stats_linear_svm(accs, task):
    print("--- Cross Validation Results:")                
    print("-- Accuracies")
    str_stats = "Task:  Avg. +- Std.  |  Max.  |  Min.\n"
    accs_array = np.array(accs)
    avg = accs_array.mean()
    std = accs_array.std()
    min = accs_array.min()
    max = accs_array.max()
    str_stats += f"{task:>4}:{avg:>7.4f}+-{std:<7.4f}|{max:^8.4f}|{min:^8.4f}\n"
    print(str_stats)

from argument_parser import parse_arguments
import contextlib
import copy
from functools import partial
import math
import os
import random
import shutil
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from tqdm import tqdm
from tqdm.auto import trange

from inner_loop_adapters import update_parameters_gd
from model import MultitaskGCN, MultitaskGCN_2
import data_utils
import test_embeddings
import utils as ut
from utils import EpochStats, BaselinesCVStats


def adapt_and_test(model, task, train_batch, test_batch, args, log_vars=None):
    """Adapt model on train_batch, and test it on test_batch. Returns statistics, inner loss,
    and outer loss (loss on test_batch with adapted parameters) that can be used for global 
    update (outer loop)."""
    train_logit = model(train_batch, task_selector=task)
    if task == "gc":
        train_targets = train_batch.y
        test_targets = test_batch.y
        
        inner_loss = F.cross_entropy(train_logit, train_targets)
        if log_vars and (args.weight_unc == 1):
            precision = torch.exp(-log_vars[task])
            inner_loss = torch.sum(precision * inner_loss + log_vars[task], -1)
            if log_vars[task].grad:
                log_vars[task].grad.zero_()
        model.zero_grad()
        adapted_params = update_parameters_gd(model, inner_loss,
            step_size=args.step_size, first_order=args.first_order)
            
        test_logit = model(test_batch, task_selector=task, params=adapted_params)
        outer_loss = F.cross_entropy(test_logit, test_targets)
        with torch.no_grad():
            test_acc = ut.get_accuracy(test_logit, test_targets)
    elif task == "nc":
        node_labels = train_batch.node_y.argmax(1)
        train_mask = train_batch.train_mask.squeeze()
        test_mask = (train_mask==0).float()
        
        inner_loss = F.cross_entropy(train_logit[train_mask==1], node_labels[train_mask==1])                             
        if log_vars and (args.weight_unc == 1):
            precision = torch.exp(-log_vars[task])
            inner_loss = torch.sum(precision * inner_loss + log_vars[task], -1)
            if log_vars[task].grad:
                log_vars[task].grad.zero_()
        model.zero_grad()
        adapted_params = update_parameters_gd(model, inner_loss,
            step_size=args.step_size, first_order=args.first_order)
        
        test_logit = model(train_batch, task_selector=task, params=adapted_params)
        outer_loss = F.cross_entropy(test_logit[test_mask==1], 
                                     node_labels[test_mask==1])
        with torch.no_grad():
            test_acc = ut.get_accuracy(test_logit[test_mask==1], node_labels[test_mask==1])
    elif task == "lp":
        train_link_labels = data_utils.get_link_labels(train_batch.pos_edge_index, train_batch.neg_edge_index)
        test_link_labels = data_utils.get_link_labels(test_batch.pos_edge_index, test_batch.neg_edge_index)
        
        inner_loss = F.binary_cross_entropy_with_logits(train_logit.squeeze(), train_link_labels)
        if log_vars and (args.weight_unc == 1):
            precision = torch.exp(-log_vars[task])
            inner_loss = torch.sum(precision * inner_loss + log_vars[task], -1)
            if log_vars[task].grad:
                log_vars[task].grad.zero_()

        model.zero_grad()
        adapted_params = update_parameters_gd(model, inner_loss,
            step_size=args.step_size, first_order=args.first_order)
        
        test_logit = model(test_batch, task_selector=task, params=adapted_params)
        outer_loss = F.binary_cross_entropy_with_logits(test_logit.squeeze(), test_link_labels)
        with torch.no_grad():
            #test_logit = torch.sigmoid(test_logit)
            test_logit = test_logit.detach().cpu().numpy()
            test_link_labels = test_link_labels.detach().cpu().numpy()
            try:
                test_acc = torch.tensor(roc_auc_score(test_link_labels, test_logit.squeeze()))
            except ValueError:
                print("Problem in AUC")
                print("Test Logit: {},\n Test Link Labels: {}".format(test_logit, test_link_labels))
                test_acc = torch.tensor(0.0)
    elif isinstance(task, list): # we are in the concurrent case
        inner_loss = {}
        if "gc" in task:
            gc_logit = train_logit["gc"]
            gc_train_targets = train_batch.y
            gc_test_targets = test_batch.y
            inner_loss["gc"] = F.cross_entropy(gc_logit, gc_train_targets)
        if "nc" in task:
            nc_logit = train_logit["nc"]
            train_node_labels = train_batch.node_y.argmax(1)
            nc_train_mask = train_batch.train_mask.squeeze()
            test_node_labels = test_batch.node_y.argmax(1)
            nc_test_mask = (test_batch.train_mask.squeeze() == 0).float()
            inner_loss["nc"] = F.cross_entropy(nc_logit[nc_train_mask==1], train_node_labels[nc_train_mask==1])                             
        if "lp" in task:
            lp_logit = train_logit["lp"]
            train_link_labels = data_utils.get_link_labels(train_batch.pos_edge_index, train_batch.neg_edge_index)
            test_link_labels = data_utils.get_link_labels(test_batch.pos_edge_index, test_batch.neg_edge_index)
            inner_loss["lp"] = F.binary_cross_entropy_with_logits(lp_logit.squeeze(), train_link_labels)

        inner_sum = torch.tensor(0.).to(args.device)
        if log_vars and (args.weight_unc == 1):
            for t in task:
                precision = torch.exp(-log_vars[t])
                inner_sum += torch.sum(precision * inner_loss[t] + log_vars[t], -1)
                if log_vars[t].grad:
                    log_vars[t].grad.zero_()
        else:
            for t in task:
                inner_sum += inner_loss[t]


        model.zero_grad()
        adapted_params = update_parameters_gd(model, inner_sum, step_size=args.step_size, first_order=args.first_order)

        test_logit = model(test_batch, task_selector=task, params=adapted_params)

        outer_loss = {}
        if "gc" in task:
            gc_test_logit = test_logit["gc"]
            outer_loss["gc"] = F.cross_entropy(gc_test_logit, gc_test_targets)
        if "nc" in task:
            nc_test_logit = test_logit["nc"]
            outer_loss["nc"] = F.cross_entropy(nc_test_logit[nc_test_mask==1], test_node_labels[nc_test_mask==1]) 
        if "lp" in task:
            lp_test_logit = test_logit["lp"]
            outer_loss["lp"] = F.binary_cross_entropy_with_logits(lp_test_logit.squeeze(), test_link_labels)

        test_acc= {}
        with torch.no_grad():
            if "gc" in task:
                test_acc["gc"] = ut.get_accuracy(gc_test_logit, gc_test_targets)
            if "nc" in task:
                test_acc["nc"] = ut.get_accuracy(nc_test_logit[nc_test_mask==1], test_node_labels[nc_test_mask==1])
            if "lp" in task:
                lp_test_logit = lp_test_logit.detach().cpu().numpy()
                test_link_labels = test_link_labels.detach().cpu().numpy()
                try:
                    test_acc["lp"] = torch.tensor(roc_auc_score(test_link_labels, lp_test_logit.squeeze()))
                except ValueError:
                    print("Problem in AUC")
                    print("Test Logit: {},\n Test Link Labels: {}".format(lp_test_logit, test_link_labels))
                    test_acc["lp"] = torch.tensor(0.0)

    return outer_loss, inner_loss, test_acc
        
        
def meta_train(model, dataloader, prepare_batch_tasks_func, args, batch_task_list=None, val_dataloader=None, val_batch_task_list=None):
    model.train()

    log_vars = None
    if args.weight_unc:
        log_var_nc = torch.zeros((1,), requires_grad=True, device=args.device)
        log_var_gc = torch.zeros((1,), requires_grad=True, device=args.device)
        log_var_lp = torch.zeros((1,), requires_grad=True, device=args.device)
        log_vars = {"nc":log_var_nc, "gc":log_var_gc, "lp":log_var_lp}
        p_list = [param for param in model.parameters()] + [log_var_nc, log_var_gc, log_var_lp]
        meta_optimizer = torch.optim.Adam(p_list, lr=args.meta_lr)
    else:
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)

    # Meta-Training loop
    global_tasks_stats = []
    if args.early_stopping:
        best_val_score = 0
        best_val_loss = 100
        if not args.es_tmpdir:
            args.es_tmpdir = f"{args.meta_alg}_{args.batch_task}_early_stopping_tmp"
    for epoch in trange(args.epochs, desc="Train Epoch"):
        epoch_stats = EpochStats()

        dataloader_2 = None
        shuffle = False
        if shuffle:
            batches = [b for b in dataloader]
            index_shuf = list(range(len(batches)))
            random.shuffle(index_shuf)
            dataloader_2 = [batches[i] for i in index_shuf]
            if batch_task_list:
                batch_task_list = [batch_task_list[i] for i in index_shuf]
        if dataloader_2:
            selected_dataloader = dataloader_2
        else:
            selected_dataloader = dataloader
        for batch_idx, batch in enumerate(tqdm(selected_dataloader, desc="Train Batch")):
            model.zero_grad()

            # Prepare Meta-Train training and test data
            if batch_task_list:
                task_for_batch = batch_task_list[batch_idx]
                tasks, train_batches, test_batches = prepare_batch_tasks_func(batch, False, task_for_batch)
            else:
                tasks, train_batches, test_batches = prepare_batch_tasks_func(batch, False)
            
            # Adapt model to Meta-Train train data, and test on Meta-Train test data
            total_outer_loss = torch.tensor(0.).to(args.device)
            for task, train_batch, test_batch in zip(tasks, train_batches, test_batches):
                train_batch = train_batch.to(args.device)
                if test_batch:
                    test_batch = test_batch.to(args.device)
                outer_loss, inner_loss, test_acc = adapt_and_test(model, task, train_batch, test_batch, args, log_vars)

                if isinstance(task, list): # we are in the concurrent case
                    for t in task:
                        epoch_stats.update(t, train_batch, test_batch, inner_loss[t], outer_loss[t], test_acc[t], True)
                        if args.weight_unc == 2:
                            precision = torch.exp(-log_vars[t])
                            total_outer_loss += torch.sum(precision * outer_loss[t] + log_vars[t], -1)
                        else:
                            total_outer_loss += outer_loss[t]
                    continue

                if args.weight_unc == 2:
                    precision = torch.exp(-log_vars[task])
                    total_outer_loss += torch.sum(precision * outer_loss + log_vars[task], -1)
                else:
                    total_outer_loss += outer_loss# / train_batch.num_graphs ########### NOW IT'S THE SUM OF THE PER-SAMPLE AVERAGES
                
                #with torch.no_grad():
                epoch_stats.update(task, train_batch, test_batch, inner_loss, outer_loss, test_acc)

            # Global Update
            #total_outer_loss.div_(args.batch_size) losses are already averaged
            total_outer_loss.backward()
            meta_optimizer.step()
        
        if args.early_stopping and epoch%50 == 0:
            model_copy = copy.deepcopy(model)
            tqdm.write("\nTest on Validation Set")
            val_stats = meta_test(model_copy, val_dataloader, prepare_batch_tasks_func, args, val_batch_task_list)
            tot_acc = 0
            tot_loss = 0
            for task in val_stats:
                tot_acc += val_stats[task]["acc"]
                tot_loss += val_stats[task]["outer_loss"]
            if tot_acc > best_val_score:
                best_val_score = tot_acc
                model_copy.to("cpu")
                args.early_stopping_stats = val_stats # so it save them in file
                args.early_stopping_tot_acc = tot_acc
                args.early_stopping_tot_loss = tot_loss
                args.early_stopping_epoch = epoch
                ut.save_model(model_copy, args.es_tmpdir, "best_val", args)
            if tot_loss < best_val_loss:
                best_val_loss = tot_loss
                model_copy.to("cpu")
                args.early_stopping_stats = val_stats # so it save them in file
                args.early_stopping_tot_acc = tot_acc
                args.early_stopping_tot_loss = tot_loss
                args.early_stopping_epoch = epoch
                ut.save_model(model_copy, args.es_tmpdir, "best_val_loss", args)

        tasks_epoch_stats = epoch_stats.get_average_stats()
        global_tasks_stats.append(tasks_epoch_stats)
        ut.print_train_epoch_stats(epoch, tasks_epoch_stats)

    if args.early_stopping:
        ut.recover_early_stopping_best_weights(model, args.es_tmpdir, name="best_val", delete_dir=False)

    return global_tasks_stats


def meta_test(model, dataloader, prepare_batch_tasks_func, args, batch_task_list=None):
    model.eval()
    # Meta-Testing loop
    epoch_stats = EpochStats()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Eval Batch")):
        model.zero_grad()

        # Prepare Meta-Test training and test data
        if batch_task_list:
            task_for_batch = batch_task_list[batch_idx]
            tasks, train_batches, test_batches = prepare_batch_tasks_func(batch, False, task_for_batch)
        else:
            tasks, train_batches, test_batches = prepare_batch_tasks_func(batch, False)
        
        # Adapt model to Meta-Test train data, and test on Meta-Test test data
        for task, train_batch, test_batch in zip(tasks, train_batches, test_batches):
            train_batch = train_batch.to(args.device)
            if test_batch:
                test_batch = test_batch.to(args.device)
            outer_loss, inner_loss, test_acc = adapt_and_test(model, task, train_batch, test_batch, args)

            if isinstance(task, list): # we are in the concurrent case
                for t in task:
                    epoch_stats.update(t, train_batch, test_batch, inner_loss[t], outer_loss[t], test_acc[t], True)
                continue

            epoch_stats.update(task, train_batch, test_batch, inner_loss, outer_loss, test_acc)
    
    tasks_test_stats = epoch_stats.get_average_stats()
    ut.print_test_stats(tasks_test_stats)
    return tasks_test_stats


def run(args):
    args.device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print("Using device:", args.device)
    
    dataset, train_val_test_ratio = data_utils.get_graph_dataset(args.dataset_name, 
                                                                 destination_dir=args.data_folder)
    
    cv_test_stats = []
    cv_test_stats_best_loss = []
    if args.test_emb:
        cv_baselines_test_stats = BaselinesCVStats()
    for cv_fold in range(args.folds):
        print(f"Cross Validation Fold: {cv_fold}", flush=True)
        dataset = dataset.shuffle()
        meta_train_dataloader, meta_val_dataloader, meta_test_dataloader = data_utils.get_dataloaders(dataset, 
                                                                                 args.batch_size,
                                                                                 args.batch_task,
                                                                                 train_val_test_ratio=train_val_test_ratio,
                                                                                 num_workers=1)

        if args.meta_alg == "MAML":
            model_func = MultitaskGCN
        elif args.meta_alg == "ANIL":
            model_func = MultitaskGCN_2
        model = model_func(dataset.num_node_features, 
                           args.embedding_dim,
                           dataset[0].node_y.size(1),
                           dataset.num_classes,
                           residual_con=args.residual_con,
                           normalize_emb=args.normalize_emb,
                           batch_norm=args.batch_norm,
                           dropout=args.dropout)
        model = model.to(args.device)
                             
        train_batch_task_list = val_batch_task_list = test_batch_task_list = None
        if args.batch_task == "single":
            train_batch_task_list = data_utils.create_batch_task_list(len(meta_train_dataloader), tasks=args.tasks)
            val_batch_task_list = data_utils.create_batch_task_list(len(meta_val_dataloader), tasks=args.tasks)
            test_batch_task_list = data_utils.create_batch_task_list(len(meta_test_dataloader), tasks=args.tasks)
            prepare_batch_tasks_func = data_utils.single_task_train_test_split
        elif args.batch_task == "multi":
            prepare_batch_tasks_func = partial(data_utils.multi_task_train_test_split, tasks=args.tasks)
        elif args.batch_task == "conc":
            prepare_batch_tasks_func = partial(data_utils.concurrent_multi_task_train_test_split, tasks=args.tasks)

        with open(os.devnull, "w") as f, contextlib.ExitStack() as gs:
            if args.folds > 1:
                    gs.enter_context(contextlib.redirect_stdout(f))
                    gs.enter_context(contextlib.redirect_stderr(f))
            global_training_stats = meta_train(model,
                                               meta_train_dataloader,
                                               prepare_batch_tasks_func,
                                               args,
                                               train_batch_task_list,
                                               meta_val_dataloader,
                                               val_batch_task_list)
    
        if args.create_training_plots:
            cv_filename_prefix = ""
            if args.folds > 1:
                cv_filename_prefix = str(cv_fold)
            ut.create_stats_plots(global_training_stats, cv_filename_prefix)
        
        if args.output_folder:
            if args.folds > 1:
                name = f"cv_{cv_fold}"
            else:
                name = "multitask_gcn"
            saved_dir = ut.save_model(model, args.output_folder, name, args)
            print("Model saved at path:", saved_dir)

        # For testing the batch size is always 6
        tasks_test_stats = meta_test(model, meta_test_dataloader, prepare_batch_tasks_func, args, test_batch_task_list)
        cv_test_stats.append(tasks_test_stats)

        ## Try also model with best val loss
        if args.early_stopping:
            model_best_loss = copy.deepcopy(model)
            ut.recover_early_stopping_best_weights(model_best_loss, args.es_tmpdir, name="best_val_loss")
            tasks_test_stats = meta_test(model_best_loss, meta_test_dataloader, prepare_batch_tasks_func, args, test_batch_task_list)
            cv_test_stats_best_loss.append(tasks_test_stats)

        if args.test_emb:
            if args.output_folder:
                folder_name = "baselines" if args.folds == 1 else f"baselines_cv_{cv_fold}"
                baselines_output_folder = os.path.join(args.output_folder, folder_name)
            else:
                baselines_output_folder = None
            with open(os.devnull, "w") as f, contextlib.ExitStack() as gs:
                print("Run Test Embeddings")
                if args.folds > 1:
                        gs.enter_context(contextlib.redirect_stdout(f))
                        gs.enter_context(contextlib.redirect_stderr(f))
                embedding_stats = test_embeddings.run_test(model,
                                                           meta_train_dataloader.dataset,
                                                           meta_val_dataloader.dataset,
                                                           meta_test_dataloader.dataset,
                                                           epochs=100,
                                                           batch_size=8,
                                                           lr=1e-3,
                                                           embedding_dim=args.embedding_dim,
                                                           hidden_dim=args.embedding_dim,
                                                           early_stopping=True,
                                                           es_tmpdir=args.es_tmpdir,
                                                           output_folder=baselines_output_folder,
                                                           device=args.device)
                cv_baselines_test_stats.update(embedding_stats) 

    print("\n\n############## Meta-Learned Multitask GCN ##############")
    print("Best Val Acc")
    ut.print_cv_stats(cv_test_stats)
    if args.early_stopping:
        print("\nBest Val Loss")
        ut.print_cv_stats(cv_test_stats_best_loss)
    if args.test_emb:
       cv_baselines_test_stats.print_stats() 

    return cv_test_stats, model
        

if __name__ == "__main__":
    args = parse_arguments("MultitaskGCN")
    ut.set_seeds()
    ut.print_arguments(args) 
    cv_stats, model = run(args)    

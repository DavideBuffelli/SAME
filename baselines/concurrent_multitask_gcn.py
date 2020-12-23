import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from argument_parser import parse_arguments
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from tqdm.auto import trange

from baselines.baseline_models import MultiTaskGCN
import baselines.bl_utils as bl_ut
from baselines.bl_utils import EpochStats
import data_utils
import test_embeddings
import utils as ut


def concurrent_multi_task_train_test_split(batch, train, tasks=["gc", "nc", "lp"]):
    if batch.num_graphs < 2:
        print("Error! Batch size too small for concurrent_multi_task_train_test_split. You need at least 2 graphs.")
        exit()
    
    list_batch = batch.to_data_list()
    
    # Graphs are already ready for Graph Classification 

    # For Node Classification, add a train mask to each node
    if "nc" in tasks:
        if train:
            nc_train_ratio = 1
        else:
            nc_train_ratio = 0
        list_batch = data_utils.prepare_data_for_node_classification(list_batch, nc_train_ratio, False)
    
    # For Link Prediction create positive and negative examples (we hide 1-lp_train_ratio % of the edges)
    if "lp" in tasks:
        lp_train_ratio = 0.8
        train_data_list, test_data_list = data_utils.prepare_data_for_link_prediction(list_batch,
                                                                    train_ratio=lp_train_ratio,
                                                                    neg_to_pos_edge_ratio=1,
                                                                    rnd_labeled_edges=False)
        if train:
            for train_data, test_data in zip(train_data_list, test_data_list):
                train_data.pos_edge_index = torch.cat((train_data.pos_edge_index, test_data.pos_edge_index), dim=1)
                train_data.neg_edge_index = torch.cat((train_data.neg_edge_index, test_data.neg_edge_index), dim=1)
            list_batch = train_data_list
        else:
            list_batch = test_data_list

    train_batch = Batch.from_data_list(list_batch)
    return ["all"], [train_batch], None
  
  
def train(model, dataloader, args, val_dataloader=False):
    model.train()
    if args.weight_unc:
        log_var_nc = torch.zeros((1,), requires_grad=True, device=args.device)
        log_var_gc = torch.zeros((1,), requires_grad=True, device=args.device)
        log_var_lp = torch.zeros((1,), requires_grad=True, device=args.device)
        log_vars = {"nc":log_var_nc, "gc":log_var_gc, "lp":log_var_lp}
        p_list = [param for param in model.parameters()] + [log_var_nc, log_var_gc, log_var_lp]
        optimizer = torch.optim.Adam(p_list, lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    if args.early_stopping:
        best_val_score = 0
        if not args.es_tmpdir:
            args.es_tmpdir = "bmt_early_stopping_tmp"
    for epoch in trange(args.epochs, desc="Epoch"):
        epoch_stats = EpochStats()
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Batch")):
            optimizer.zero_grad()
            
            _, train_batch, _ = concurrent_multi_task_train_test_split(batch, True, tasks=args.tasks)
            train_batch = train_batch[0]
            train_batch = train_batch.to(args.device)
            
            # Forward pass
            gc_train_logit, nc_train_logit, lp_train_logit = model(train_batch)
            
            # Evaluate Loss and Accuracy
            # GC
            gc_loss = nc_loss = lp_loss = 0
            if "gc" in args.tasks:
                gc_loss = F.cross_entropy(gc_train_logit, train_batch.y)
                with torch.no_grad():
                    gc_acc = ut.get_accuracy(gc_train_logit, train_batch.y)
                epoch_stats.update("gc", train_batch, gc_loss, gc_acc, True)
            # NC
            if "nc" in args.tasks:
                node_labels = train_batch.node_y.argmax(1)
                train_mask = train_batch.train_mask.squeeze()
                nc_loss = F.cross_entropy(nc_train_logit[train_mask==1], node_labels[train_mask==1])
                with torch.no_grad():
                    nc_acc = ut.get_accuracy(nc_train_logit[train_mask==1], node_labels[train_mask==1])
                epoch_stats.update("nc", train_batch, nc_loss, nc_acc, True)
            # LP
            if "lp" in args.tasks:
                train_link_labels = data_utils.get_link_labels(train_batch.pos_edge_index, train_batch.neg_edge_index)
                lp_loss = F.binary_cross_entropy_with_logits(lp_train_logit.squeeze(), train_link_labels)
                with torch.no_grad():
                    train_labels = train_link_labels.detach().cpu().numpy()
                    train_predictions = lp_train_logit.detach().cpu().numpy()
                    lp_acc = roc_auc_score(train_labels, train_predictions.squeeze())
                epoch_stats.update("lp", train_batch, lp_loss, lp_acc, True)

            if args.weight_unc:
                gc_precision = torch.exp(-log_vars["gc"]) if "gc" in args.tasks else 0
                nc_precision = torch.exp(-log_vars["nc"]) if "nc" in args.tasks else 0
                lp_precision = torch.exp(-log_vars["lp"]) if "lp" in args.tasks else 0
                loss = torch.sum(gc_precision * gc_loss + log_vars["gc"], -1) + \
                       torch.sum(nc_precision * nc_loss + log_vars["nc"], -1) + \
                       torch.sum(lp_precision * lp_loss + log_vars["lp"], -1)
            else:
                loss = gc_loss + nc_loss + lp_loss
            
            # Backprop and update parameters
            loss.backward()
            optimizer.step()

        if args.early_stopping and epoch%10 == 0:
            model_copy = copy.deepcopy(model)
            tqdm.write("\nTest on Validation Set")
            val_stats = test(model_copy, val_dataloader, args)
            tot_acc = 0
            for task in val_stats:
                tot_acc += val_stats[task]["acc"]
            if tot_acc > best_val_score:
                best_val_score = tot_acc
                model_copy.to("cpu")
                args.early_stopping_stats = val_stats
                args.early_stopping_tot_acc = tot_acc
                args.early_stopping_epoch = epoch
                ut.save_model(model_copy, args.es_tmpdir, "best_val", args)

        tasks_epoch_stats = epoch_stats.get_average_stats()
        bl_ut.print_train_epoch_stats(epoch, tasks_epoch_stats)

    if args.early_stopping:
        ut.recover_early_stopping_best_weights(model, args.es_tmpdir)


def test(model, dataloader, args):
    model.eval()
    epoch_stats = EpochStats()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Batch")):
        _, test_batch, _ = concurrent_multi_task_train_test_split(batch, False, tasks=args.tasks)
        test_batch = test_batch[0]
        test_batch = test_batch.to(args.device)
        with torch.no_grad():
            gc_test_logit, nc_test_logit, lp_test_logit = model(test_batch)
            # GC
            if "gc" in args.tasks:
                gc_loss = F.cross_entropy(gc_test_logit, test_batch.y)
                with torch.no_grad():
                    gc_acc = ut.get_accuracy(gc_test_logit, test_batch.y)
                epoch_stats.update("gc", test_batch, gc_loss, gc_acc, False)
            #NC
            if "nc" in args.tasks:
                node_labels = test_batch.node_y.argmax(1)
                train_mask = test_batch.train_mask.squeeze()
                test_mask = (train_mask==0).float()
                nc_loss = F.cross_entropy(nc_test_logit[test_mask==1], node_labels[test_mask==1])
                with torch.no_grad():
                    nc_acc = ut.get_accuracy(nc_test_logit[test_mask==1], node_labels[test_mask==1])
                epoch_stats.update("nc", test_batch, nc_loss, nc_acc, False)
            # LP
            if "lp" in args.tasks:
                test_link_labels = data_utils.get_link_labels(test_batch.pos_edge_index, test_batch.neg_edge_index)
                lp_loss = F.binary_cross_entropy_with_logits(lp_test_logit.squeeze(), test_link_labels)
                with torch.no_grad():
                    test_labels = test_link_labels.detach().cpu().numpy()
                    test_predictions = lp_test_logit.detach().cpu().numpy()
                    lp_acc = roc_auc_score(test_labels, test_predictions.squeeze())
                epoch_stats.update("lp", test_batch, lp_loss, lp_acc, False)

    tasks_test_stats = epoch_stats.get_average_stats()
    bl_ut.print_test_stats(tasks_test_stats)
    return tasks_test_stats


def run(args):
    args.device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print("Using device:", args.device)

    dataset, train_val_test_ratio = data_utils.get_graph_dataset(args.dataset_name, 
                                                                 destination_dir=args.data_folder)
    
    cv_test_stats = []
    if args.test_emb:
        cv_baselines_test_stats = ut.BaselinesCVStats() 
    for cv_fold in range(args.folds):
        print(f"Cross Validation Fold: {cv_fold}")
        dataset = dataset.shuffle()
        train_dataloader, val_dataloader, test_dataloader = data_utils.get_dataloaders(dataset, 
                                                                               args.batch_size,
                                                                               "multi",
                                                                               train_val_test_ratio=[0.7, 0.1, 0.2],
                                                                               num_workers=1,
                                                                               shuffle_train=True)

        output_gc_dim = dataset.num_classes
        output_nc_dim = dataset[0].node_y.size(1)
        model = MultiTaskGCN(args.tasks,
                             dataset.num_node_features, 
                             args.embedding_dim,
                             output_gc_dim,
                             output_nc_dim,
                             residual_con=args.residual_con,
                             normalize_emb=args.normalize_emb,
                             batch_norm=args.batch_norm,
                             dropout=args.dropout)
        model = model.to(args.device)

        train(model, train_dataloader, args, val_dataloader)

        if args.output_folder:
            if args.folds > 1:
                name = f"concurrent_multitask_gcn_cv_{cv_fold}"
            else:
                name = "concurrent_multitask_gcn"
            saved_dir = ut.save_model(model, args.output_folder, name, args)
            print("Model saved at path:", saved_dir)
    
        test_acc = test(model, test_dataloader, args)
        cv_test_stats.append(test_acc)

        if args.test_emb:
            if args.output_folder:
                folder_name = "baselines" if args.folds == 1 else f"baselines_cv_{cv_fold}"
                baselines_output_folder = os.path.join(args.output_folder, folder_name)
            else:
                baselines_output_folder = None
            embedding_stats = test_embeddings.run_test(model,
                                                       train_dataloader.dataset,
                                                       val_dataloader.dataset,
                                                       test_dataloader.dataset,
                                                       epochs=100,
                                                       batch_size=16,
                                                       lr=1e-3,
                                                       embedding_dim=args.embedding_dim,
                                                       es_tmpdir=args.es_tmpdir,
                                                       hidden_dim=args.embedding_dim,
                                                       early_stopping=True,
                                                       output_folder=baselines_output_folder,
                                                       device=args.device)
            cv_baselines_test_stats.update(embedding_stats) 

    print("\n\n############## Baseline Multitask GCN ##############")
    ut.print_cv_stats(cv_test_stats)
    if args.test_emb:
        cv_baselines_test_stats.print_stats()

    return cv_test_stats, model

        
if __name__ == "__main__":
    args = parse_arguments("ConcurrentMultiTaskGCN")
    ut.set_seeds()
    ut.print_arguments(args)
    cv_stats, model = run(args)

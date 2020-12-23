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

from baselines.baseline_models import SingleTaskGCN
import baselines.bl_utils as bl_ut
from baselines.bl_utils import EpochStats
import data_utils
import test_embeddings
import utils as ut

  
def prepare_batch_for_task(batch, task, train):
    if task == "gc":
        train_ratio = 1
        list_batch = batch.to_data_list()
        train_list, _ = data_utils.prepare_data_for_graph_classification(list_batch, train_ratio, False)
        ready_batch = Batch.from_data_list(train_list)
    elif task == "nc":
        if train:
            train_ratio = 1
        else:
            train_ratio = 0
        list_batch = batch.to_data_list()
        list_batch = data_utils.prepare_data_for_node_classification(list_batch, train_ratio, False)
        ready_batch = Batch.from_data_list(list_batch)
    elif task == "lp":
        train_ratio = 0.8
        list_batch = batch.to_data_list()
        train_data_list, test_data_list = data_utils.prepare_data_for_link_prediction(list_batch, 
                                                                                      train_ratio=train_ratio,
                                                                                      neg_to_pos_edge_ratio=1,
                                                                                      rnd_labeled_edges=False)
        if train:
            for train_data, test_data in zip(train_data_list, test_data_list):
                train_data.pos_edge_index = torch.cat((train_data.pos_edge_index, test_data.pos_edge_index), dim=1)
                train_data.neg_edge_index = torch.cat((train_data.neg_edge_index, test_data.neg_edge_index), dim=1)
                #train_data.lp_labels = data_utils.get_link_labels(train_data.pos_edge_index, train_data.neg_edge_index)
            ready_batch = Batch.from_data_list(train_data_list)
        else:
            ready_batch = Batch.from_data_list(test_data_list)
    return ready_batch
          
        
def train(model, dataloader, args, val_dataloader=None):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    if args.early_stopping:
        best_val_score = 0
        if not args.es_tmpdir:
            args.es_tmpdir = args.task+"_bst_early_stopping_tmp"
    for epoch in trange(args.epochs, desc="Epoch"):
        epoch_stats = EpochStats()
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Batch")):
            optimizer.zero_grad()
            
            train_batch = prepare_batch_for_task(batch, args.task, train=True)
            train_batch = train_batch.to(args.device)

            # Forward pass 
            train_logit = model(train_batch)
            
            # Evaluate Loss and Accuracy
            if args.task == "gc":
                loss = F.cross_entropy(train_logit, train_batch.y)
                with torch.no_grad():
                    acc = ut.get_accuracy(train_logit, train_batch.y)
            elif args.task == "nc":
                node_labels = train_batch.node_y.argmax(1)
                train_mask = train_batch.train_mask.squeeze()
                loss = F.cross_entropy(train_logit[train_mask==1], node_labels[train_mask==1])
                with torch.no_grad():
                    acc = ut.get_accuracy(train_logit[train_mask==1], node_labels[train_mask==1])
            elif args.task == "lp":
                train_link_labels = data_utils.get_link_labels(train_batch.pos_edge_index, train_batch.neg_edge_index)
                loss = F.binary_cross_entropy_with_logits(train_logit.squeeze(), train_link_labels)
                with torch.no_grad():
                    train_labels = train_link_labels.detach().cpu().numpy()
                    train_predictions = train_logit.detach().cpu().numpy()
                    try:
                        acc = roc_auc_score(train_labels, train_predictions.squeeze())
                    except ValueError:
                        auc = 0.0

            epoch_stats.update(args.task, train_batch, loss, acc, True)
            
            # Backprop and update parameters
            loss.backward()
            optimizer.step()
            
        if args.early_stopping and epoch%10 == 0:
            model_copy = copy.deepcopy(model)
            tqdm.write("\nTest on Validation Set")
            val_stats = test(model_copy, val_dataloader, args)
            epoch_acc = val_stats[args.task]["acc"]
            if epoch_acc > best_val_score:
                best_val_score = epoch_acc
                model_copy.to("cpu")
                args.early_stopping_stats = val_stats
                args.early_stopping_epoch_acc = epoch_acc
                args.early_stopping_epoch = epoch
                ut.save_model(model_copy, args.es_tmpdir, "best_val", args)

        task_epoch_stats = epoch_stats.get_average_stats()
        bl_ut.print_train_epoch_stats(epoch, task_epoch_stats)

    if args.early_stopping:
        ut.recover_early_stopping_best_weights(model, args.es_tmpdir)


def test(model, dataloader, args):
    model.eval()
    epoch_stats = EpochStats()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Batch")):
        test_batch = prepare_batch_for_task(batch, args.task, train=False)
        test_batch = test_batch.to(args.device)
        with torch.no_grad():
            test_logit = model(test_batch)
            if args.task == "gc":
                loss = F.cross_entropy(test_logit, test_batch.y)
                with torch.no_grad():
                    acc = ut.get_accuracy(test_logit, test_batch.y)
            elif args.task == "nc":
                node_labels = test_batch.node_y.argmax(1)
                train_mask = test_batch.train_mask.squeeze()
                test_mask = (train_mask==0).float()
                loss = F.cross_entropy(test_logit[test_mask==1], node_labels[test_mask==1])
                with torch.no_grad():
                    acc = ut.get_accuracy(test_logit[test_mask==1], node_labels[test_mask==1])
            elif args.task == "lp":
                test_link_labels = data_utils.get_link_labels(test_batch.pos_edge_index, test_batch.neg_edge_index)
                loss = F.binary_cross_entropy_with_logits(test_logit.squeeze(), test_link_labels)
                with torch.no_grad():
                    test_labels = test_link_labels.detach().cpu().numpy()
                    test_predictions = test_logit.detach().cpu().numpy()
                    acc = roc_auc_score(test_labels, test_predictions.squeeze())

            epoch_stats.update(args.task, test_batch, loss, acc, False)

    task_test_stats = epoch_stats.get_average_stats()
    bl_ut.print_test_stats(task_test_stats)
    return task_test_stats


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
                                                                               "single",
                                                                               train_val_test_ratio=[0.7, 0.1, 0.2],
                                                                               num_workers=1,
                                                                               shuffle_train=True)

        if args.task == "gc":
            output_dim = dataset.num_classes
        elif args.task == "nc":
            output_dim = dataset[0].node_y.size(1)
        elif args.task == "lp":
            output_dim = 1
        model = SingleTaskGCN(args.task,
                              dataset.num_node_features, 
                              args.embedding_dim,
                              output_dim,
                              residual_con=args.residual_con,
                              normalize_emb=args.normalize_emb,
                              batch_norm=args.batch_norm,
                              dropout=args.dropout)
        model = model.to(args.device)

        train(model, train_dataloader, args, val_dataloader)

        if args.output_folder:
            if args.folds > 1:
                name = f"singletask_{args.task}_gcn_cv_{cv_fold}"
            else:
                name = "singletask_gcn"
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

    print("\n\n############## Single Task GCN ##############")
    ut.print_cv_stats(cv_test_stats)
    if args.test_emb:
        cv_baselines_test_stats.print_stats()

    return cv_test_stats, model
        

if __name__ == "__main__":
    args = parse_arguments("SingleTaskGCN")
    ut.set_seeds()
    ut.print_arguments(args)
    cv_stats, model = run(args)

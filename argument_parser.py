import argparse


def get_base_parser(name):
    parser = argparse.ArgumentParser(name)
    
    parser.add_argument("-dataset-name", type=str,
        help="Name of the dataset from the TUDortmund library to run code on.")
    parser.add_argument("--data-folder", type=str, default="",
        help="Path to the folder where data will be stored (default is working directory).")

    parser.add_argument("--output-folder", type=str, default=None,
        help="Path to the output folder for saving the model (optional).")
    parser.add_argument("--es-tmpdir", type=str, default=None,
        help="Path to the temporary folder for early stopping (optional).")
  
    parser.add_argument("--folds", type=int, default=1,
        help="Number of cross-validation folds (default: 1).")
    parser.add_argument("--epochs", type=int, default=10,
        help="Number of meta-learning epochs (default: 10).")
    parser.add_argument("--early-stopping", action="store_true",
        help="During training test on validation set, and return best model.")
    parser.add_argument("--batch-size", type=int, default=16,
        help="Number of tasks in a mini-batch of tasks (default: 16).")
    parser.add_argument("--embedding-dim", type=int, default=16,
        help="Node embedding dimension (default: 16).")
    parser.add_argument("--dropout", action="store_true",
        help="Use dropout inside the network when training.")
    parser.add_argument("--residual-con", action="store_true",
        help="Use residual connections in between GCN layers")
    parser.add_argument("--normalize-emb", action="store_true",
        help="Normalize node embedding to unit norm in between GCN layers")
    parser.add_argument("--batch-norm", action="store_true",
        help="Use batch normalization on node embeddings between GCN layers.")

    parser.add_argument("--test-emb", action="store_true",
        help="After training test the embeddings on multiple tasks using baseline models.")
       
    parser.add_argument("--use-cuda", action="store_true",
        help="Use CUDA if available.")

    return parser


def add_arguments_for_meta_learning(parser):
    parser.add_argument("--create-training-plots", action="store_true",
        help="Create a 'fig' folder with plots showing training stats over epochs.")

    parser.add_argument("--tasks", type=str, default="gc,nc,lp",
        help="Tasks to be performed (default is 'gc,nc,lp').")
    parser.add_argument("--meta-alg", type=str, default="MAML",
        help="Meta-Learning algorithm to use ('MAML', 'ANIL').")
    parser.add_argument("--batch-task", type=str, default="single",
        help="Functions to use to divide tasks in batch ('single'=every batch contains examples of only 1 task, 'multi'=every batch contain examples of all the tasks).")
    parser.add_argument("--first-order", action="store_true",
        help="Use the first-order approximation for the meta-update.")
    parser.add_argument("--step-size", type=float, default=0.4,
        help="Step-size for the gradient step for adaptation (default: 0.4).")
    parser.add_argument("--meta-lr", type=float, default=1e-3,
        help="Learning rate for the meta-learner (default: 1e-3).")

    parser.add_argument("--weight-unc", type=int, default=0,
        help="Weigth the multitask loss function using uncertainty (0->no; 1->on inner loss; 2->on outer loss.")


def add_arguments_for_multitask_baseline(parser):
    parser.add_argument("--tasks", type=str, default="gc,nc,lp",
        help="Tasks to be performed (default is 'gc,nc,lp').")
    parser.add_argument("--lr", type=float, default=1e-3,
        help="Learning rate for Adam Optimizer (default: 1e-3).")
    parser.add_argument("--weight-unc", action="store_true",
        help="Weigth the multitask loss function using uncertainty.")


def add_arguments_for_singletask_baseline(parser):
    parser.add_argument("-task", type=str,
        help="Task to perform (one of: 'gc', 'nc', 'lp').")
    parser.add_argument("--lr", type=float, default=1e-3,
        help="Learning rate for Adam Optimizer (default: 1e-3).")


def parse_arguments(name):
    parser = get_base_parser(name)
    if name == "MultitaskGCN":
        add_arguments_for_meta_learning(parser)
    elif name == "ConcurrentMultiTaskGCN":
        add_arguments_for_multitask_baseline(parser)
    elif name == "SingleTaskGCN":
        add_arguments_for_singletask_baseline(parser)
    args = parser.parse_args()
    if hasattr(args, "tasks"):
        args.tasks = list(args.tasks.split(","))
    return args

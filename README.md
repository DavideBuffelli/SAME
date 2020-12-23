# A Meta-Learning Approach for Graph Representation Learning in Multi-Task Settings

Reference code for the paper: "A Meta-Learning Approach for Graph Representation Learning in Multi-Task Settings", presentedat the NeurIPS Workshop on Meta-Learning (MetaLearn) 2020.
Please cite the paper if you use this code in your own work. 
```
@article{buffellimetalearn2020,
  author = {Buffelli, Davide and Vandin, Fabio},
  title = {A Meta-Learning Approach for Graph Representation Learning in Multi-Task Settings},
  journal = {NeurIPS Workshop on Meta-Learning (MetaLearn)},
  year = {2020}
}
```

Link to paper: <https://arxiv.org/abs/2012.06755>

## Instructions
To launch the training of our proposed method on a dataset from the [TUD](https://chrsmrrs.github.io/datasets/)
collection you can use:

``` python train.py -dataset-name INSERT_DATASET_NAME --INSERT_OPTIONS```

To get a list of all the possible options and their description:

```python train.py -h```

### Reproducing the Results of the Paper
#### Single Task Baseline
```
cd baselines
python one_task_gcn.py -dataset-name ENZYMES -task gc --batch-size 32 --embedding-dim 256 --epochs 1000 --lr 0.00043609137255698444 --residual-con --use-cuda --early-stopping --test-emb --folds 10
python one_task_gcn.py -dataset-name ENZYMES -task nc --batch-size 64 --embedding-dim 256 --epochs 1000 --lr 0.019219461332261562 --residual-con --use-cuda --early-stopping --test-emb --folds 10
python one_task_gcn.py -dataset-name ENZYMES -task lp --batch-size 64 --embedding-dim 256 --epochs 1000 --lr 0.0014400268897414208 --normalize-emb --use-cuda --early-stopping --test-emb --folds 10
python one_task_gcn.py -dataset-name PROTEINS -task gc --batch-size 32 --embedding-dim 256 --epochs 1000 --lr 0.00043609137255698444 --residual-con --use-cuda --early-stopping --test-emb --folds 10
python one_task_gcn.py -dataset-name PROTEINS -task nc --batch-size 64 --embedding-dim 256 --epochs 1000 --lr 0.019219461332261562 --residual-con --use-cuda --early-stopping --test-emb --folds 10
python one_task_gcn.py -dataset-name PROTEINS -task lp --batch-size 64 --embedding-dim 256 --epochs 1000 --lr 0.0014400268897414208 --normalize-emb --use-cuda --early-stopping --test-emb --folds 10
python one_task_gcn.py -dataset-name COX2 -task gc --batch-size 16 --embedding-dim 128 --epochs 1000 --lr 0.005 --use-cuda --early-stopping --test-emb --folds 10
python one_task_gcn.py -dataset-name COX2 -task nc --batch-size 16 --embedding-dim 256 --epochs 1000 --lr 0.0964517825126648 --residual-con --normalize-emb --batch-norm  --use-cuda --early-stopping --test-emb --folds 10
python one_task_gcn.py -dataset-name COX2 -task lp --batch-size 64 --embedding-dim 256 --epochs 1000 --lr 0.0002495164792984724 --batch-norm --use-cuda --early-stopping --test-emb --folds 10
python one_task_gcn.py -dataset-name DHFR -task gc --batch-size 16 --embedding-dim 128 --epochs 1000 --lr 0.005 --use-cuda --early-stopping --test-emb --folds 10
python one_task_gcn.py -dataset-name DHFR -task nc --batch-size 16 --embedding-dim 256 --epochs 1000 --lr 0.0964517825126648 --residual-con --normalize-emb --batch-norm  --use-cuda --early-stopping --test-emb --folds 10
python one_task_gcn.py -dataset-name DHFR -task lp --batch-size 64 --embedding-dim 256 --epochs 1000 --lr 0.0002495164792984724 --batch-norm --use-cuda --early-stopping --test-emb --folds 10
```

#### Multi-task baseline
```
cd baselines
python concurrent_multitask_gcn.py -dataset-name PROTEINS --tasks gc,nc --batch-size 64 --embedding-dim 256 --epochs 5000 --lr 0.01 --folds 10 --residual-con --normalize-emb --use-cuda --early-stopping --test-emb 
python concurrent_multitask_gcn.py -dataset-name PROTEINS --tasks gc,lp --batch-size 64 --embedding-dim 256 --epochs 5000 --folds 10 --lr 0.01 --residual-con --normalize-emb --use-cuda --early-stopping --test-emb
python concurrent_multitask_gcn.py -dataset-name PROTEINS --tasks nc,lp --batch-size 64 --embedding-dim 256 --epochs 5000 --lr 0.01 --folds 10 --residual-con --normalize-emb --use-cuda --early-stopping --test-emb
python concurrent_multitask_gcn.py -dataset-name PROTEINS --batch-size 64 --embedding-dim 256 --epochs 5000 --lr 0.01 --folds 10 --residual-con --normalize-emb --use-cuda --early-stopping --test-emb
python concurrent_multitask_gcn.py -dataset-name ENZYMES --tasks gc,nc --batch-size 64 --embedding-dim 256 --epochs 5000 --lr 0.01 --folds 10 --residual-con --normalize-emb --use-cuda --early-stopping --test-emb 
python concurrent_multitask_gcn.py -dataset-name ENZYMES --tasks gc,lp --batch-size 64 --embedding-dim 256 --epochs 5000 --folds 10 --lr 0.01 --residual-con --normalize-emb --use-cuda --early-stopping --test-emb
python concurrent_multitask_gcn.py -dataset-name ENZYMES --tasks nc,lp --batch-size 64 --embedding-dim 256 --epochs 5000 --lr 0.01 --folds 10 --residual-con --normalize-emb --use-cuda --early-stopping --test-emb
python concurrent_multitask_gcn.py -dataset-name ENZYMES --batch-size 64 --embedding-dim 256 --epochs 5000 --lr 0.01 --folds 10 --residual-con --normalize-emb --use-cuda --early-stopping --test-emb
python concurrent_multitask_gcn.py -dataset-name COX2 --tasks gc,nc --batch-size 64 --embedding-dim 512 --epochs 5000 --lr 0.09 --folds 10 --residual-con --batch-norm --use-cuda --early-stopping --test-emb
python concurrent_multitask_gcn.py -dataset-name COX2 --tasks gc,lp --batch-size 64 --embedding-dim 512 --epochs 5000 --lr 0.09 --folds 10  --residual-con --batch-norm --use-cuda --early-stopping --test-emb
python concurrent_multitask_gcn.py -dataset-name COX2 --tasks nc,lp --batch-size 64 --embedding-dim 512 --epochs 5000 --lr 0.09 --folds 10 --residual-con --batch-norm --use-cuda --early-stopping --test-emb
python concurrent_multitask_gcn.py -dataset-name COX2 --batch-size 64 --embedding-dim 512 --epochs 5000 --lr 0.09 --folds 10 --residual-con --batch-norm --use-cuda --early-stopping --test-emb
python concurrent_multitask_gcn.py -dataset-name DHFR --tasks gc,nc --batch-size 64 --embedding-dim 512 --epochs 5000 --lr 0.09 --folds 10 --residual-con --batch-norm --use-cuda --early-stopping --test-emb
python concurrent_multitask_gcn.py -dataset-name DHFR --tasks gc,lp --batch-size 64 --embedding-dim 512 --epochs 5000 --lr 0.09 --folds 10  --residual-con --batch-norm --use-cuda --early-stopping --test-emb
python concurrent_multitask_gcn.py -dataset-name DHFR --tasks nc,lp --batch-size 64 --embedding-dim 512 --epochs 5000 --lr 0.09 --folds 10 --residual-con --batch-norm --use-cuda --early-stopping --test-emb
python concurrent_multitask_gcn.py -dataset-name DHFR --batch-size 64 --embedding-dim 512 --epochs 5000 --lr 0.09 --folds 10 --residual-con --batch-norm --use-cuda --early-stopping --test-emb
```

#### SAME
We report the instructions for iSAME on the ENZYMES dataset. To launch on another dataset change "ENZYMES" with "PROTEINS", "DHFR", or "COX2".
Change "MAML" with "ANIL" if you want to use eSAME instead.
```
python train.py -dataset-name ENZYMES --batch-size 16 --embedding-dim 256 --epochs 8000 --meta-lr 0.0005 --step-size 0.04 --folds 10 --residual-con --normalize-emb --early-stopping --batch-task multi --tasks gc,nc --meta-alg MAML --test-emb --use-cuda
python train.py -dataset-name ENZYMES --batch-size 16 --embedding-dim 256 --epochs 8000 --meta-lr 0.0005 --step-size 0.04 --folds 10 --residual-con --normalize-emb --early-stopping --batch-task multi --tasks gc,lp --meta-alg MAML --test-emb --use-cuda
python train.py -dataset-name ENZYMES --batch-size 16 --embedding-dim 256 --epochs 8000 --meta-lr 0.0005 --step-size 0.04 --folds 10 --residual-con --normalize-emb --early-stopping --batch-task multi --tasks nc,lp --meta-alg MAML --test-emb --use-cuda 
python train.py -dataset-name ENZYMES --batch-size 16 --embedding-dim 256 --epochs 15000 --meta-lr 0.0005 --step-size 0.04 --folds 10 --residual-con --normalize-emb --early-stopping --batch-task multi --meta-alg MAML --test-emb --use-cuda 
```

## Requirements

This code requires Python 3.6 (or higher) and makes use of the following packages:

* PyTorch 1.4.0
* PyTorch Geometric 1.4.3 [(Installation Instructions)](https://github.com/rusty1s/pytorch_geometric)
* Matplotlib 3.2.1
* NetworkX 2.4
* scikit-learn 0.22.2
* tqdm 4.45.0

## License
Refer to the file [LICENSE](LICENSE)

import pytorch_lightning as pl
import sys
from matminer.featurizers.site import *
import matminer

site_feauturizers_dict = matminer.featurizers.site.__dict__
import yaml
from pytorch_lightning.callbacks import *
import argparse
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import torch
import pandas as pd
from scipy import stats
import numpy as np
import sys, os
from tqdm import tqdm
import torch
from torch import nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression,ElasticNet
import pickle as pk
import matminer
from matminer.featurizers.structure.composite import JarvisCFID
from matminer.featurizers.structure.misc import XRDPowderPattern
from multiprocessing import Pool
from h5_handler import *
import os
os.environ["export MKL_NUM_THREADS"] = "1"
os.environ["export NUMEXPR_NUM_THREADS"] = "1"
os.environ["export OMP_NUM_THREADS"] = "1"
os.environ["export OPENBLAS_NUM_THREADS"] = "1"
from matminer.featurizers.structure.bonding import BagofBonds
from matminer.featurizers.structure.matrix import OrbitalFieldMatrix,SineCoulombMatrix
#monkeypatches

compression_alg = "gzip"

import pickle as pk

class DIM_h5_Data_Module(pl.LightningDataModule):
    def __init__(
        self,
        config,
        overwrite=False,
        ignore_errors=False,
        max_len=100,
        Dataset=None,
        cpus = 1,
        chunk_size = 32,
        multitaskmode_labels=False,
        seed="FIXED_SEED",
        **kwargs
    ):

        super().__init__()
        self.seed = seed
        self.batch_size = config["Batch_Size"]
        #In dynamic batching, the number of unique sites is the limit on the batch, not the number of crystals, the number of crystals varies between batches
        self.dynamic_batch = config["dynamic_batch"]
        self.Site_Features = config["Site_Features"]
        self.Site_Labels = config["Site_Labels"]
        self.Interaction_Features = config["Interaction_Features"]
        self.h5_file = config["h5_file"]
        self.overwrite = overwrite
        self.ignore_errors = ignore_errors
        self.limit = config["Max_Samples"]
        self.max_len = max_len
        self.cpus=cpus
        self.multitaskmode_labels = multitaskmode_labels
        if Dataset is None:
            self.Dataset = torch_h5_cached_loader(
                self.Site_Features,
                self.Site_Labels,
                self.Interaction_Features,
                self.h5_file,
                max_len=self.max_len,
                ignore_errors=self.ignore_errors,
                overwrite=self.overwrite,
                limit=self.limit,
                cpus=cpus,
                chunk_size=chunk_size,
                seed=self.seed
            )
        else:
            self.Dataset = Dataset

    def prepare_data(self):
        self.Dataset_Train, self.Dataset_Val = random_split(
            self.Dataset,
            [len(self.Dataset) - len(self.Dataset) // 20, len(self.Dataset) // 20],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        torch.manual_seed(hash(self.seed))#Makes sampling reproducable
        if self.dynamic_batch:
            if self.multitaskmode_labels:
                return DataLoader(
                    self.Dataset_Train,
                    collate_fn=collate_fn,
                    batch_sampler=Multitask_batch_sampler(SequentialSampler(self.Dataset_Train),self.batch_size,N_labels=self.multitaskmode_labels),
                    pin_memory=False,
                    num_workers=self.cpus,
                    prefetch_factor=8,
                    persistent_workers=True
                )
            else:
                return DataLoader(
                    self.Dataset_Train,
                    collate_fn=collate_fn,
                    batch_sampler=SiteNet_batch_sampler(RandomSampler(self.Dataset_Train),self.batch_size),
                    pin_memory=False,
                    num_workers=self.cpus,
                    prefetch_factor=8,
                    persistent_workers=True
                )
        else:
            return DataLoader(
                self.Dataset_Train,
                batch_size=self.batch_size,
                collate_fn=collate_fn,
                pin_memory=False,
                num_workers=self.cpus,
                prefetch_factor=8,
                persistent_workers=True
            )

    def val_dataloader(self):
        torch.manual_seed(hash(self.seed))#Makes sampling reproducable
        if self.dynamic_batch:
            return DataLoader(
                self.Dataset_Val,
                collate_fn=collate_fn,
                batch_sampler=SiteNet_batch_sampler(RandomSampler(self.Dataset_Val),self.batch_size),
                pin_memory=False,
                num_workers=self.cpus,
                prefetch_factor=8,
                persistent_workers=True
            )
        else:
            return DataLoader(
                self.Dataset_Val,
                batch_size=self.batch_size,
                collate_fn=collate_fn,
                pin_memory=False,
                num_workers=self.cpus,
                prefetch_factor=8,
                persistent_workers=True
            )

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="ml options")
    parser.add_argument("-w", "--number_of_worker_processes", default=1,type=int)
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("-u", "--cell_size_limit", default = None )
    parser.add_argument("-l", "--load_from_cache", default = False)
    parser.add_argument("-f","--featurizer",default=None)
    parser.add_argument("-d","--dataset",default=None)

    datasets = {"egap":{"train":"Data/Matbench/matbench_mp_gap_cubic_50_train_1.hdf5","test":"Data/Matbench/matbench_mp_gap_cubic_50_test_1.hdf5"},
                "eform":{"train":"Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5","test":"Data/Matbench/matbench_mp_e_form_cubic_50_test_1.hdf5"}}
    
    featurizers = {"orbital":OrbitalFieldMatrix(period_tag=True),
                   "jarvis":JarvisCFID(use_cell=False,use_chg=False,use_adf=False,use_ddf=False,use_nn=False),
                   "xray":XRDPowderPattern()}

    args = parser.parse_args()
    args.cell_size_limit = int(args.cell_size_limit)

    limits = [10,50,100,250,1000,10000]
    repeats = [100,100,100,25,10,5]

    results_dataframe = pd.DataFrame(columns = ["rf_R2","rf_MAE","rf_MSE","nn_R2","nn_MAE","nn_MSE","lin_R2","lin_MAE","lin_MSE","model","limit","measure"])

    with open(str(args.config), "r") as config_file:
        config = yaml.load(config_file)
    config["dynamic_batch"] = False
    config["Batch_Size"] = 128
    config["Max_Samples"] = 1000000
    config["h5_file"] = datasets[args.dataset]["train"]
    if not bool(args.load_from_cache):
        train_data = DIM_h5_Data_Module(
                config,
                max_len=args.cell_size_limit,
                ignore_errors=True,
                overwrite=False,
                cpus=args.number_of_worker_processes,
                chunk_size=32,
            )
    config["Max_Samples"] = 1000000
    config["h5_file"] = datasets[args.dataset]["test"]
    if not bool(args.load_from_cache):
        test_data = DIM_h5_Data_Module(
                config,
                max_len=args.cell_size_limit,
                ignore_errors=True,
                overwrite=False,
                cpus=args.number_of_worker_processes,
                chunk_size=32,
            )
    
    results_dataframe = pd.DataFrame(columns = ["rf_R2","rf_MAE","rf_MSE","nn_R2","nn_MAE","nn_MSE","lin_R2","lin_MAE","lin_MSE","model","limit","measure"])

    class wrapped_featurizer():
        def __init__(self,featurizer):
            self.featurizer = featurizer
        def featurize(self,structure):
            try:
                return self.featurizer.featurize(structure)
            except Exception as E:
                print(E)
                return None
    
    featurizer = wrapped_featurizer(OrbitalFieldMatrix(period_tag=True))

    if bool(args.load_from_cache):
        print("Loading Training Data From Cache")
        file = open("featurizer_pickles/" + args.dataset + args.featurizer + '_feat_features_downstream_train_cache.pk', 'rb')
        training_features_full = pk.load(file)
        file = open("featurizer_pickles/" + args.dataset + args.featurizer + '_feat_labels_downstream_train_cache.pk', 'rb')
        training_labels_full = pk.load(file)
    else:
        print("Featurizing Training Data")
        pool = Pool(32)
        training_features_full = pool.map(featurizer.featurize,tqdm([i["structure"] for i in train_data.Dataset]))
        pool.close()
        pool.join()
        training_labels_full = np.array([i["target"] for i,j in zip(train_data.Dataset,training_features_full) if j is not None])
        training_features_full = np.array([i for i in training_features_full if i is not None])
        file = open("featurizer_pickles/" + args.dataset + args.featurizer + '_feat_features_downstream_train_cache.pk', 'wb')
        pk.dump(training_features_full,file)
        file = open("featurizer_pickles/" + args.dataset + args.featurizer + '_feat_labels_downstream_train_cache.pk', 'wb')
        pk.dump(training_labels_full,file)

    if bool(args.load_from_cache):
        print("Loading Test Data From Cache")
        file = open("featurizer_pickles/" + args.dataset + args.featurizer + '_feat_features_downstream_test_cache.pk', 'rb')
        test_features_full = pk.load(file)
        file = open("featurizer_pickles/" + args.dataset + args.featurizer + '_feat_labels_downstream_test_cache.pk', 'rb')
        test_labels_full = pk.load(file)
    else:
        print("Featurizing Test Data")
        pool = Pool(32)
        test_features_full = pool.map(featurizer.featurize,tqdm([i["structure"] for i in test_data.Dataset]))
        pool.close()
        pool.join()
        test_labels_full = np.array([i["target"] for i,j in zip(test_data.Dataset,test_features_full) if j is not None])
        test_features_full = np.array([i for i in test_features_full if i is not None])
        file = open("featurizer_pickles/" + args.dataset + args.featurizer + '_feat_features_downstream_test_cache.pk', 'wb')
        pk.dump(test_features_full,file)
        file = open("featurizer_pickles/" + args.dataset + args.featurizer + '_feat_labels_downstream_test_cache.pk', 'wb')
        pk.dump(test_labels_full,file)


    print("Featurized!")

    for limit,iterations in zip(limits,repeats):
        training_features_samples = [np.random.choice(np.arange(len(training_features_full)), size=min(limit,len(training_features_full)), replace=False) for i in range(iterations)]
        rows = pd.DataFrame(columns = ["rf_R2","rf_MAE","rf_MSE","nn_R2","nn_MAE","nn_MSE","lin_R2","lin_MAE","lin_MSE"])
        for i,sample in enumerate(training_features_samples):
            print(i)

            rf = RandomForestRegressor().fit(training_features_full[sample,:], training_labels_full[sample])
            nn = MLPRegressor(hidden_layer_sizes=64, max_iter=5000).fit(training_features_full[sample,:], training_labels_full[sample])
            lin = LinearRegression().fit(training_features_full[sample,:], training_labels_full[sample])

            rows = rows.append(pd.DataFrame({
                "rf_R2": rf.score(test_features_full, test_labels_full),
                "rf_MAE":np.mean(np.absolute(rf.predict(test_features_full)-test_labels_full)),
                "rf_MSE":np.mean(np.array(rf.predict(test_features_full)-test_labels_full)**2),
                "nn_R2": nn.score(test_features_full, test_labels_full),
                "nn_MAE":np.mean(np.absolute(nn.predict(test_features_full)-test_labels_full)),
                "nn_MSE":np.mean(np.array(nn.predict(test_features_full)-test_labels_full)**2),
                "lin_R2": lin.score(test_features_full, test_labels_full),
                "lin_MAE":np.mean(np.absolute(lin.predict(test_features_full)-test_labels_full)),
                "lin_MSE":np.mean(np.array(lin.predict(test_features_full)-test_labels_full)**2),
            },
            index=[str(i)]))
        
        rows["model"] = args.dataset + "_" + args.featurizer
        rows["limit"] = limit
        results_dataframe = results_dataframe.append(rows, ignore_index=True)
        print(results_dataframe)
        results_dataframe.to_csv("featurizer_pickles/" + args.dataset + args.featurizer + "_Downstream_Featurized.csv")  

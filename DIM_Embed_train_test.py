import pytorch_lightning as pl
import sys
from matminer.featurizers.site import *
import matminer

site_feauturizers_dict = matminer.featurizers.site.__dict__
from lightning_module import (
    basic_callbacks,
    DIM_h5_Data_Module,
    SiteNet,
    SiteNet_DIM
)
from lightning_module import basic_callbacks
import yaml
from pytorch_lightning.callbacks import *
import argparse
import os
os.environ["export MKL_NUM_THREADS"] = "1"
os.environ["export NUMEXPR_NUM_THREADS"] = "1"
os.environ["export OMP_NUM_THREADS"] = "1"
os.environ["export OPENBLAS_NUM_THREADS"] = "1"
import torch
import pandas as pd
from scipy import stats
import numpy as np
import sys, os
from modules import SiteNetAttentionBlock,SiteNetEncoder,k_softmax
from tqdm import tqdm
from lightning_module import collate_fn
from lightning_module import af_dict as lightning_af_dict
import torch
from torch_scatter import segment_coo,segment_csr
from torch import nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression,ElasticNet
import pickle as pk
#monkeypatches

compression_alg = "gzip"

import pickle as pk

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="ml options")
    parser.add_argument("-w", "--number_of_worker_processes", default=1,type=int)
    parser.add_argument("-u", "--cell_size_limit", default = None )
    args = parser.parse_args()
    config_and_model = [["Initial_eform","config/compact_dim_klnorm.yaml",None,"e_form"],
                        #["klnorm_multiloss","config/compact_dim_klnorm.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_klnorm_DIM-v2.ckpt","e_form"],
                        #["nocomp_klnorm_multiloss","config/compact_dim_nocomp_klnorm.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_nocomp_klnorm_DIM.ckpt","e_form"],
                        ["nocomp_klnorm_moremultiloss_eform","config/compact_dim_nocomp_klnorm.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_nocomp_klnorm_DIM-v2.ckpt","e_form"],
                        ["Initial_egap","config/compact_dim_klnorm.yaml",None,"e_gap"],
                        ["nocomp_klnorm_moremultiloss_egap","config/compact_dim_nocomp_klnorm.yaml","Data/Matbench/matbench_mp_gap_cubic_50_train_1.hdf5_best_compact_dim_nocomp_klnorm_DIM.ckpt","e_gap"],
                        #["nonorm","config/compact_dim_nonorm.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_nonorm_DIM-v2.ckpt","e_form"],
                        #["nocomp","config/compact_dim_nocomp.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_nocomp_DIM.ckpt","e_form"],
                        ["klnorm_eform","config/compact_dim_klnorm.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_klnorm_DIM.ckpt","e_form"],
                        #["doublenorm","config/compact_dim_doublenorm.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_doublenorm_DIM.ckpt","e_form"],
                        #["adversarialnorm","config/compact_dim_adversarialnorm.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_adversarialnorm_DIM.ckpt","e_form"],
                        #["localonly","config/compact_dim_localonly.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_localonly_DIM.ckpt","e_form"],
                        #["onlycomp","config/compact_dim_onlycomp.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_onlycomp_DIM.ckpt","e_form"],
                        #["control","config/compact_dim_nothing.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_nothing_DIM.ckpt","e_form"],
                        #["control_cutoff5","config/compact_dim_nothing_cutoff5.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_nothing_cutoff5_DIM-v2.ckpt","e_form"],
                        #["nonorm_cutoff5","config/compact_dim_nonorm_cutoff5.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_nonorm_cutoff5_DIM.ckpt","e_form"],
                        ]

    #config_and_model = [["klnorm_multiloss","config/compact_dim_klnorm.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_klnorm_DIM-v2.ckpt"]]

    limits = [10, 50, 100, 250, 1000, 10000]
    repeats = [100, 100, 100, 25, 10, 5]

    results_dataframe = pd.DataFrame(columns = ["rf_R2","rf_MAE","rf_MSE","nn_R2","nn_MAE","nn_MSE","lin_R2","lin_MAE","lin_MSE","model","limit","measure"])

    train_data_dict = {"e_form":"Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5","e_gap":"Data/Matbench/matbench_mp_gap_cubic_50_train_1.hdf5"}
    test_data_dict = {"e_form":"Data/Matbench/matbench_mp_e_form_cubic_50_test_1.hdf5","e_gap":"Data/Matbench/matbench_mp_gap_cubic_50_test_1.hdf5"}

    for cm in config_and_model:
        print("Model type is " + cm[0])

        torch.set_num_threads(args.number_of_worker_processes)
        try:
            print("config file is " + cm[1])
            with open(str(cm[1]), "r") as config_file:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
        except Exception as e:
            raise RuntimeError(
                "Config not found or unprovided, a path to a configuration yaml must be provided with -c"
            )
        results_list = []
        model_name = cm[2]
        dataset_name = train_data_dict[cm[3]] #Get train dataset according to training target
        #config["Max_Samples"] = 100
        config["h5_file"] = dataset_name
        config["dynamic_batch"] = False
        config["Batch_Size"] = 128
        if args.cell_size_limit != None:
            args.cell_size_limit = int(args.cell_size_limit)
        Dataset = DIM_h5_Data_Module(
            config,
            max_len=args.cell_size_limit,
            ignore_errors=True,
            overwrite=False,
            cpus=args.number_of_worker_processes,
            chunk_size=32,
        )

        dataset_name = test_data_dict[cm[3]] #Get test dataset according to training target
        config["h5_file"] = dataset_name
        Dataset_Test = DIM_h5_Data_Module(
            config,
            max_len=args.cell_size_limit,
            ignore_errors=True,
            overwrite=False,
            cpus=args.number_of_worker_processes,
            chunk_size=32
        )

        #for limit,repeat in zip(limits,repeats):
        torch.cuda.empty_cache()
        model = SiteNet_DIM(config)
        model.to(device)
        if model_name != None:
            print("DIM PARAMETERS")
            model.load_state_dict(torch.load(model_name,map_location=torch.device("cpu"))["state_dict"], strict=True)
        else:
            print("INITIAL PARAMETERS")
        results = model.forward(Dataset.Dataset,batch_size=128).detach().cpu().numpy()
        results_y = np.array([Dataset.Dataset[i]["target"] for i in range(len(Dataset.Dataset))])
        results_Test = model.forward(Dataset_Test.Dataset,batch_size=128).detach().cpu().numpy()
        results_test_y = np.array([Dataset_Test.Dataset[i]["target"] for i in range(len(Dataset_Test.Dataset))])
        for limit,repeat in zip(limits,repeats):
            print("Limit is " + str(limit))
            samples = [np.random.choice(np.arange(len(Dataset.Dataset)), size=min(limit,len(Dataset.Dataset)), replace=False) for i in range(repeat)]
            rows = pd.DataFrame(columns = ["rf_R2","rf_MAE","rf_MSE","nn_R2","nn_MAE","nn_MSE","lin_R2","lin_MAE","lin_MSE"])
            for i,sample in enumerate(samples):
                rf = RandomForestRegressor().fit(results[sample,:], results_y[sample])
                nn = MLPRegressor(hidden_layer_sizes=64, max_iter=5000).fit(results[sample,:], results_y[sample])
                lin = LinearRegression().fit(results[sample,:], results_y[sample])

                rows = rows.append(pd.DataFrame({
                    "rf_R2": rf.score(results_Test, results_test_y),
                    "rf_MAE":np.mean(np.absolute(rf.predict(results_Test)-results_test_y)),
                    "rf_MSE":np.mean(np.array(rf.predict(results_Test)-results_test_y)**2),
                    "nn_R2": nn.score(results_Test, results_test_y),
                    "nn_MAE":np.mean(np.absolute(nn.predict(results_Test)-results_test_y)),
                    "nn_MSE":np.mean(np.array(nn.predict(results_Test)-results_test_y)**2),
                    "lin_R2": lin.score(results_Test, results_test_y),
                    "lin_MAE":np.mean(np.absolute(lin.predict(results_Test)-results_test_y)),
                    "lin_MSE":np.mean(np.array(lin.predict(results_Test)-results_test_y)**2),
                },
                index=[str(i)]))
            
            rows["model"] = cm[0]
            rows["limit"] = limit
            results_dataframe = results_dataframe.append(rows, ignore_index=True)
            results_dataframe.to_csv("Downstream_DIM.csv")  


""" 
            stds = rows.std(ddof=0)
            stds["model"] = cm[0]
            stds["limit"] = limit
            stds["measure"] = "STD"
            results_dataframe = results_dataframe.append(stds, ignore_index=True)

            medians = rows.median()
            medians["model"] = cm[0]
            medians["limit"] = limit
            medians["measure"] = "Median"
            results_dataframe = results_dataframe.append(medians, ignore_index=True)

            mins = rows.min()
            mins["model"] = cm[0]
            mins["limit"] = limit
            mins["measure"] = "Min"
            results_dataframe = results_dataframe.append(mins, ignore_index=True)

            maxes = rows.max()
            maxes["model"] = cm[0]
            maxes["limit"] = limit
            maxes["measure"] = "Max"
            results_dataframe = results_dataframe.append(maxes, ignore_index=True)

            lowerquantile = rows.quantile(q=0.25)
            lowerquantile["model"] = cm[0]
            lowerquantile["limit"] = limit
            lowerquantile["measure"] = "LowQ"
            results_dataframe = results_dataframe.append(lowerquantile, ignore_index=True)

            upperquantile = rows.quantile(q=0.75)
            upperquantile["model"] = cm[0]
            upperquantile["limit"] = limit
            upperquantile["measure"] = "UpQ"
            results_dataframe = results_dataframe.append(upperquantile, ignore_index=True)
"""
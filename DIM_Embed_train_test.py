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
    parser.add_argument("-f", "--h5_train_file_name", default=None)
    parser.add_argument("-t", "--h5_test_file_name")
    parser.add_argument("-w", "--number_of_worker_processes", default=1,type=int)
    parser.add_argument("-u", "--cell_size_limit", default = None )
    parser.add_argument("-r", "--repeats", default = 1,type=int )
    args = parser.parse_args()
    config_and_model = [["klnorm_multiloss","config/compact_dim_klnorm.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_klnorm_DIM-v2.ckpt"],
                        ["nonorm","config/compact_dim_nonorm.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_nonorm_DIM-v2.ckpt"],
                        ["nocomp","config/compact_dim_nocomp.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_nocomp_DIM.ckpt"],
                        ["klnorm","config/compact_dim_klnorm.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_klnorm_DIM.ckpt"],
                        ["doublenorm","config/compact_dim_doublenorm.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_doublenorm_DIM.ckpt"],
                        ["adversarialnorm","config/compact_dim_adversarialnorm.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_adversarialnorm_DIM.ckpt"],
                        ["localonly","config/compact_dim_localonly.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_localonly_DIM.ckpt"],
                        #["onlycomp","config/compact_dim_onlycomp.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_onlycomp_DIM.ckpt"],
                        #["control","config/compact_dim_nothing.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_nothing_DIM.ckpt"],
                        #["control_cutoff5","config/compact_dim_nothing_cutoff5.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_nothing_cutoff5_DIM-v2.ckpt"],
                        #["nonorm_cutoff5","config/compact_dim_nonorm_cutoff5.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_nonorm_cutoff5_DIM.ckpt"],
                        ]

    #config_and_model = [["klnorm_multiloss","config/compact_dim_klnorm.yaml","Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_klnorm_DIM-v2.ckpt"]]

    limits = [100, 250, 1000, 10000]

    results_dataframe = pd.DataFrame(columns = ["rf_R2","rf_MAE","rf_MSE","nn_R2","nn_MAE","nn_MSE","lin_R2","lin_MAE","lin_MSE","model","limit","schema"])

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
        if args.h5_train_file_name == None:
            raise RuntimeError(
                "train h5 file path is None, h5 file path must be provided through -f"
            )
        if args.h5_test_file_name == None:
            raise RuntimeError(
                "test h5 file path is None, h5 file path must be provided through -f"
            )
        results_list = []
        model_name = cm[2]
        dataset_name = args.h5_train_file_name
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

        dataset_name = args.h5_test_file_name
        config["h5_file"] = dataset_name
        Dataset_Test = DIM_h5_Data_Module(
            config,
            max_len=args.cell_size_limit,
            ignore_errors=True,
            overwrite=False,
            cpus=args.number_of_worker_processes,
            chunk_size=32
        )

        for limit in limits:
            torch.cuda.empty_cache()
            print("Limit is " + str(limit))

            rows = pd.DataFrame(columns = ["rf_R2","rf_MAE","rf_MSE","nn_R2","nn_MAE","nn_MSE","lin_R2","lin_MAE","lin_MSE"])
            samples = [np.random.choice(np.arange(len(Dataset.Dataset)), size=min(limit,len(Dataset.Dataset)), replace=False) for i in range(args.repeats)]
            for i,sample in enumerate(samples):
                model = SiteNet_DIM(config)
                model.to(device)
                results = model.forward([Dataset.Dataset[i] for i in sample],batch_size=128).detach().cpu().numpy()
                results_y = np.array([Dataset.Dataset[i]["target"] for i in sample])

                results_Test = model.forward(Dataset_Test.Dataset,batch_size=128).detach().cpu().numpy()
                results_test_y = np.array([Dataset_Test.Dataset[i]["target"] for i in range(len(Dataset_Test.Dataset))])

                rf = RandomForestRegressor().fit(results, results_y)
                nn = MLPRegressor(hidden_layer_sizes=64, max_iter=5000).fit(results, results_y)
                lin = LinearRegression().fit(results, results_y)

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

            means = rows.mean()
            means["model"] = cm[0]
            means["limit"] = limit
            means["schema"] = "Initial"
            means["measure"] = "Mean"
            results_dataframe = results_dataframe.append(means, ignore_index=True)

            stds = rows.std(ddof=0)
            stds["model"] = cm[0]
            stds["limit"] = limit
            stds["schema"] = "Initial"
            stds["measure"] = "STD"
            results_dataframe = results_dataframe.append(stds, ignore_index=True)

            medians = rows.median()
            medians["model"] = cm[0]
            medians["limit"] = limit
            medians["schema"] = "Initial"
            medians["measure"] = "Median"
            results_dataframe = results_dataframe.append(medians, ignore_index=True)

            mins = rows.min()
            mins["model"] = cm[0]
            mins["limit"] = limit
            mins["schema"] = "Initial"
            mins["measure"] = "Min"
            results_dataframe = results_dataframe.append(mins, ignore_index=True)

            maxes = rows.max()
            maxes["model"] = cm[0]
            maxes["limit"] = limit
            maxes["schema"] = "Initial"
            maxes["measure"] = "Max"
            results_dataframe = results_dataframe.append(maxes, ignore_index=True)

            lowerquantile = rows.quantile(q=0.25)
            lowerquantile["model"] = cm[0]
            lowerquantile["limit"] = limit
            lowerquantile["schema"] = "Initial"
            lowerquantile["measure"] = "LowQ"
            results_dataframe = results_dataframe.append(lowerquantile, ignore_index=True)

            upperquantile = rows.quantile(q=0.75)
            upperquantile["model"] = cm[0]
            upperquantile["limit"] = limit
            upperquantile["schema"] = "Initial"
            upperquantile["measure"] = "UpQ"
            results_dataframe = results_dataframe.append(upperquantile, ignore_index=True)

            model = SiteNet_DIM(config)
            model.to(device)
            model.load_state_dict(torch.load(model_name,map_location=torch.device("cpu"))["state_dict"], strict=True)
            results = model.forward(Dataset.Dataset,batch_size=128).detach().cpu().numpy()
            results_y = np.array([Dataset.Dataset[i]["target"] for i in range(len(Dataset.Dataset))])
            results_Test = model.forward(Dataset_Test.Dataset,batch_size=128).detach().cpu().numpy()
            results_test_y = np.array([Dataset_Test.Dataset[i]["target"] for i in range(len(Dataset_Test.Dataset))])

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
            
            means = rows.mean()
            means["model"] = cm[0]
            means["limit"] = limit
            means["schema"] = "DIM"
            means["measure"] = "Mean"
            results_dataframe = results_dataframe.append(means, ignore_index=True)

            stds = rows.std(ddof=0)
            stds["model"] = cm[0]
            stds["limit"] = limit
            stds["schema"] = "DIM"
            stds["measure"] = "STD"
            results_dataframe = results_dataframe.append(stds, ignore_index=True)

            medians = rows.median()
            medians["model"] = cm[0]
            medians["limit"] = limit
            medians["schema"] = "DIM"
            medians["measure"] = "Median"
            results_dataframe = results_dataframe.append(medians, ignore_index=True)

            mins = rows.min()
            mins["model"] = cm[0]
            mins["limit"] = limit
            mins["schema"] = "DIM"
            mins["measure"] = "Min"
            results_dataframe = results_dataframe.append(mins, ignore_index=True)

            maxes = rows.max()
            maxes["model"] = cm[0]
            maxes["limit"] = limit
            maxes["schema"] = "DIM"
            maxes["measure"] = "Max"
            results_dataframe = results_dataframe.append(maxes, ignore_index=True)

            lowerquantile = rows.quantile(q=0.25)
            lowerquantile["model"] = cm[0]
            lowerquantile["limit"] = limit
            lowerquantile["schema"] = "DIM"
            lowerquantile["measure"] = "LowQ"
            results_dataframe = results_dataframe.append(lowerquantile, ignore_index=True)

            upperquantile = rows.quantile(q=0.75)
            upperquantile["model"] = cm[0]
            upperquantile["limit"] = limit
            upperquantile["schema"] = "DIM"
            upperquantile["measure"] = "UpQ"
            results_dataframe = results_dataframe.append(upperquantile, ignore_index=True)

            results_dataframe.to_csv("Downstream_DIM.csv")  

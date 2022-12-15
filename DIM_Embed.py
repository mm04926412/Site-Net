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
import numpy as np
import sys, os
from modules import SiteNetAttentionBlock,SiteNetEncoder,k_softmax
from tqdm import tqdm
from lightning_module import collate_fn
from lightning_module import af_dict as lightning_af_dict
from torch_scatter import segment_coo,segment_csr
from torch import nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pickle as pk
#monkeypatches

compression_alg = "gzip"

import pickle as pk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ml options")
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("-f", "--h5_file_name", default=None)
    parser.add_argument("-n", "--limit", default=None,type=int)
    parser.add_argument("-m", "--model_name", default=None,type=str)
    parser.add_argument("-w", "--number_of_worker_processes", default=1,type=int)
    args = parser.parse_args()
    torch.set_num_threads(args.number_of_worker_processes)
    try:
        print("config file is " + args.config)
        with open(str(args.config), "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
    except Exception as e:
        raise RuntimeError(
            "Config not found or unprovided, a path to a configuration yaml must be provided with -c"
        )
    if args.h5_file_name == None:
        raise RuntimeError(
            "h5 file path is None, h5 file path must be provided through -f"
        )
    results_list = []
    model_name = args.model_name
    dataset_name = args.h5_file_name
    config["h5_file"] = dataset_name
    config["Max_Samples"] = args.limit
    config["dynamic_batch"] = False
    config["Batch_Size"] = 128
    Dataset = DIM_h5_Data_Module(
        config,
        max_len=None,
        ignore_errors=True,
        overwrite=False,
        cpus=args.number_of_worker_processes,
        chunk_size=32
    ) 

    model = SiteNet_DIM(config)
    results = model.forward(Dataset.Dataset,batch_size=128)
    results = pd.DataFrame(results.detach().numpy())
    tsne =np.transpose(TSNE().fit_transform(results))
    results.to_csv("embeddings_initial.csv")
    model = RandomForestRegressor()
    y = [Dataset.Dataset[i]["target"] for i in range(len(Dataset.Dataset))]
    model.fit(results[:int(len(results)*(3/4))], y[:int(len(results)*(3/4))])
    print("Initial Parameters")
    print(model.score(results[int(len(results)*3/4):], y[int(len(results)*3/4):]))
    print(np.mean(np.absolute(model.predict(results[int(len(results)*3/4):])-y[int(len(results)*3/4):])))

    plt.figure()
    plt.scatter(tsne[0],tsne[1],c=y)
    plt.savefig("tsne_initial.png")

    print(Dataset.Dataset[0].keys())
    results["Structure"] = [i["structure"] for i in Dataset.Dataset]
    pk.dump(results,open("embeddings_with_structures_initial.pk","wb"))

    model = SiteNet_DIM(config)
    model.load_state_dict(torch.load(model_name,map_location=torch.device("cpu"))["state_dict"], strict=False)
    results = model.forward(Dataset.Dataset,batch_size=128)
    results = pd.DataFrame(results.detach().numpy())
    tsne =np.transpose(TSNE().fit_transform(results))
    results.to_csv("embeddings.csv")
    model = RandomForestRegressor()
    y = [Dataset.Dataset[i]["target"] for i in range(len(Dataset.Dataset))]
    model.fit(results[:int(len(results)*(3/4))], y[:int(len(results)*(3/4))])
    print("DIM")
    print(model.score(results[int(len(results)*3/4):], y[int(len(results)*3/4):]))
    print(np.mean(np.absolute(model.predict(results[int(len(results)*3/4):])-y[int(len(results)*3/4):])))

    plt.figure()
    plt.scatter(tsne[0],tsne[1],c=y)
    plt.savefig("tsne.png")

    print(Dataset.Dataset[0].keys())
    results["Structure"] = [i["structure"] for i in Dataset.Dataset]
    pk.dump(results,open("embeddings_with_structures.pk","wb"))








    

import pytorch_lightning as pl
import sys
from matminer.featurizers.site import *
import matminer

site_feauturizers_dict = matminer.featurizers.site.__dict__
from lightning_module import (
    basic_callbacks,
    DIM_h5_Data_Module,
    SiteNet,
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
from lightning_module import collate_fn,SiteNet_DIM_supervisedcontrol,SiteNet_DIM
from lightning_module import af_dict as lightning_af_dict
from torch_scatter import segment_coo,segment_csr
from torch import nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#monkeypatches
class TReLU(torch.autograd.Function):
    """
    A transparent version of relu that has a linear gradient but sets negative values to zero,
     used as the last step in band gap prediction to provide an alternative to relu which does not kill gradients
      but also prevents the model from being punished for negative band gap predictions as these can readily be interpreted as zero
    """

    @staticmethod
    def forward(ctx, input):
        """
        f(x) is equivalent to relu
        """
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        f'(x) is linear
        """
        return grad_output


import pickle as pk

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="ml options")
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("-n", "--limit", default=None,type=int)
    #parser.add_argument("-m", "--model_name", default=None,type=str)
    parser.add_argument("-w", "--number_of_worker_processes", default=1,type=int)
    parser.add_argument("-u", "--cell_size_limit", default = None )
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
    results_list = []
    #model_name = args.model_name
    config["h5_file"] = "Data/Matbench/matbench_mp_e_form_cubic_50_test_1.hdf5"
    config["Max_Samples"] = args.limit
    config["dynamic_batch"] = False
    config["Batch_Size"] = 128
    model = SiteNet_DIM(config).to(device)
    test_scores = {}

    if args.cell_size_limit != None:
            args.cell_size_limit = int(args.cell_size_limit)

    if os.path.exists("Downstream_MAEs.csv"):
        existing_data = (pd.read_csv("Downstream_MAEs.csv",index_col=0,squeeze=True,header=None))
    else:
        existing_data = pd.Series()

    Dataset_eform = DIM_h5_Data_Module(
            config,
            max_len=args.cell_size_limit,
            ignore_errors=True,
            overwrite=False,
            cpus=args.number_of_worker_processes,
            chunk_size=32
        )
    
    config["h5_file"] = "Data/Matbench/matbench_mp_gap_cubic_50_test_1.hdf5"
    
    Dataset_egap = DIM_h5_Data_Module(
            config,
            max_len=args.cell_size_limit,
            ignore_errors=True,
            overwrite=False,
            cpus=args.number_of_worker_processes,
            chunk_size=32
        )

    DIM_model_dict = {"DIM_egap":["Data/Matbench/matbench_mp_gap_cubic_50_train_1.hdf5_best_compact_dim_nocomp_klnorm_DIM.ckpt",Dataset_egap],
                      "DIM_eform":["Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_nocomp_klnorm_DIM-v2.ckpt",Dataset_eform],
                      "random_egap":[None,Dataset_egap],
                      "random_eform":[None,Dataset_eform]}

    for key in DIM_model_dict.keys():
        if DIM_model_dict[key][0]:
            print(DIM_model_dict[key][0] + " Loaded")
            model.load_state_dict(torch.load(DIM_model_dict[key][0],map_location=torch.device("cpu"))["state_dict"], strict=False)
        else:
            model = SiteNet_DIM(config).to(device)
        embedding = pd.DataFrame(model.forward(DIM_model_dict[key][1].Dataset).detach().cpu().numpy())

        tsne =np.transpose(TSNE(init="pca",perplexity=1000,learning_rate="auto",n_iter=100000).fit_transform(embedding))
        y = [DIM_model_dict[key][1].Dataset[i]["target"] for i in range(len(DIM_model_dict[key][1].Dataset))]
        transition_metal = [DIM_model_dict[key][1].Dataset[i]["structure"].composition.contains_element_type("transition_metal") for i in range(len(DIM_model_dict[key][1].Dataset))]
        noble_gas = [DIM_model_dict[key][1].Dataset[i]["structure"].composition.contains_element_type("noble_gas") for i in range(len(DIM_model_dict[key][1].Dataset))]
        metal = [DIM_model_dict[key][1].Dataset[i]["structure"].composition.contains_element_type("metal") for i in range(len(DIM_model_dict[key][1].Dataset))]
        halogen = [DIM_model_dict[key][1].Dataset[i]["structure"].composition.contains_element_type("halogen") for i in range(len(DIM_model_dict[key][1].Dataset))]

        plt.figure()
        plt.scatter(tsne[0],tsne[1],c=y,s=0.7)
        plt.axis("off")
        plt.colorbar()
        plt.savefig("embedding_exploration/property_tsne" + key + ".png")

        plt.figure()
        plt.scatter(tsne[0],tsne[1],c=transition_metal,s=0.7)
        plt.axis("off")
        plt.savefig("embedding_exploration/transition_metal_tsne" + key + ".png")

        plt.figure()
        plt.scatter(tsne[0],tsne[1],c=noble_gas,s=0.7)
        plt.axis("off")
        plt.savefig("embedding_exploration/noble_gas_tsne" + key + ".png")

        plt.figure()
        plt.scatter(tsne[0],tsne[1],c=metal,s=0.7)
        plt.axis("off")
        plt.savefig("embedding_exploration/metal_tsne" + key + ".png")

        plt.figure()
        plt.scatter(tsne[0],tsne[1],c=halogen,s=0.7)
        plt.axis("off")
        plt.savefig("embedding_exploration/halogen_tsne" + key + ".png")
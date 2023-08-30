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
from lightning_module import collate_fn,SiteNet_DIM_supervisedcontrol
from lightning_module import af_dict as lightning_af_dict
from torch_scatter import segment_coo,segment_csr
from torch import nn
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
    model = SiteNet_DIM_supervisedcontrol(config).to(device)
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
    trained_parameters = os.listdir("Data/Matbench/downstream_models/")
    trained_parameters.sort()
    results_dictionary = {}

    for parameters in trained_parameters:
        if parameters in existing_data.index:
            print(parameters + " Already Computed")
            test_scores[parameters] = existing_data[parameters]
        else:
            try:
                print(parameters)
                model.load_state_dict(torch.load("Data/Matbench/downstream_models/" + parameters,map_location=torch.device("cpu"))["state_dict"], strict=False)
                print(parameters)
                if "egap" == parameters[:4]:
                    print("Running on egap")
                    results = model.forward(Dataset_egap.Dataset,return_truth=True,batch_size=128)
                elif "eform" == parameters[:5]:
                    print("Running on eform")
                    results = model.forward(Dataset_eform.Dataset,return_truth=True,batch_size=128)
                else:
                    raise Exception("Invalid dataset selection")
                predictions = results[0].cpu().numpy().flatten()
                truth = results[1].cpu().numpy().flatten()
                MAE = np.abs(truth-predictions).flatten().mean()
                results_dictionary[parameters] = MAE
                test_scores[parameters] = MAE
            except Exception as e:
                print(e)
                print(parameters + " failed")
    
    pd.Series(test_scores).to_csv("Downstream_MAEs.csv",header=False)



    

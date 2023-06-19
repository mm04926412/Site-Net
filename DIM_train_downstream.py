import pytorch_lightning as pl
import sys
from matminer.featurizers.site import *
import matminer
site_feauturizers_dict = matminer.featurizers.site.__dict__
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from lightning_module import (
    basic_callbacks,
    DIM_h5_Data_Module,
    SiteNet,
    SiteNet_DIM,
    SiteNet_DIM_supervisedcontrol
)
from lightning_module import basic_callbacks
import yaml
from h5_handler import torch_h5_cached_loader
from pytorch_lightning.callbacks import *
import argparse
from compress_pickle import dump, load
import collections.abc as container_abcs
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

compression_alg = "gzip"

def train_model(config, Dataset):
    model = SiteNet_DIM_supervisedcontrol(config=config, freeze=args.freeze)
    if args.starting_model != None:
        model.load_state_dict(torch.load(DIM_model_dict[args.starting_model],map_location=torch.device("cpu"))["state_dict"], strict=False)
        for layer in model.last_layer:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    if int(args.load_checkpoint) == 1:
        print(config["h5_file"])
        resume_from_checkpoint = "Data/Matbench/downstream_models/" + args.fold_name + "_" + str(config["label"]) + ".ckpt"
    #elif args.starting_model != None:
    #   print("Initializing from checkpoint " + DIM_model_dict[args.starting_model])
    #   resume_from_checkpoint = DIM_model_dict[args.starting_model]
    else:
        resume_from_checkpoint = None
    checkpoint_callback = ModelCheckpoint(
    monitor="avg_val_loss_task",
    dirpath="",
    filename="Data/Matbench/downstream_models/" + args.fold_name + "_" + "best_" + str(config["label"]),
    save_top_k=1,
    mode="min",
)
    trainer = pl.Trainer(
        gpus=int(args.num_gpus),
        callbacks=[
            basic_callbacks(filename="Data/Matbench/downstream_models/" + args.fold_name + "_"  + str(config["label"])),
            checkpoint_callback
        ],
        **config["Trainer kwargs"],
        auto_select_gpus=True,
        detect_anomaly=False,
        #gradient_clip_algorithm="value",
        log_every_n_steps=10000,
        val_check_interval=1.0,
        precision=16,
        #amp_level="O2",
        resume_from_checkpoint=resume_from_checkpoint,
    )
    trainer.fit(model, Dataset)


import pickle as pk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ml options")
    parser.add_argument("-c", "--config", default="test")
    parser.add_argument("-p", "--pickle", default=0)
    parser.add_argument("-l", "--load_checkpoint", default=0)
    parser.add_argument("-g", "--num_gpus", default=1)
    parser.add_argument("-f", "--fold_name", default="null")
    parser.add_argument("-o", "--overwrite", default=False)
    parser.add_argument("-d", "--debug", default=False)
    parser.add_argument("-u", "--unit_cell_limit",default = 100)
    parser.add_argument("-w", "--number_of_worker_processes", default=1,type=int)
    parser.add_argument("-s", "--dataseed", default="FIXED_SEED")

    parser.add_argument("-m", "--starting_model", default=None)
    parser.add_argument("-r", "--freeze", default="Neither")

    args = parser.parse_args()

    dataset = {"eform":"Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5",
               "egap":"Data/Matbench/matbench_mp_gap_cubic_50_train_1.hdf5",
               "phonons":"Data/Matbench/matbench_phonons_cubic_500_train_1.hdf5"}

    DIM_model_dict = {"DIM_egap":"Data/Matbench/matbench_mp_gap_cubic_50_train_1.hdf5_best_compact_dim_nocomp_klnorm_DIM.ckpt",
                      "DIM_eform":"Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5_best_compact_dim_nocomp_klnorm_DIM-v2.ckpt",
                      "Supervised_egap":"Data/Matbench/downstream_models/egap_best_compact_dim_sall_initial_downstream_Neither_1.ckpt",
                      "Supervised_eform":"Data/Matbench/downstream_models/eform_best_compact_dim_sall_initial_downstream_Neither_1.ckpt"}

    if args.freeze not in ["Both","Local","Global","Neither"]:
        raise Exception("Invalid Freeze Argument")
    try:
        print(args.config)
        with open(str(args.config), "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)
        raise RuntimeError(
            "Config not found or unprovided, a configuration JSON path is REQUIRED to run"
        )
    config["h5_file"] = dataset[args.fold_name]
    if args.starting_model != None:
        config["label"] = config["label"] + "_DIM_downstream_" + args.freeze + "_" + args.starting_model  + "_" + args.dataseed
    else:
        config["label"] = config["label"] + "_initial_downstream_" + args.freeze + "_" + args.dataseed
    if bool(args.debug) == True:
        config["Max_Samples"] = 30
    if int(args.pickle) == 1:
        print("Loading Pickle")
        Dataset = load(open("db_pickle.pk", "rb"), compression=compression_alg)
        Dataset.batch_size = config["Batch_Size"]
        print("Pickle Loaded")
        print("--------------")
    else:
        Dataset = DIM_h5_Data_Module(
            config,
            max_len=int(args.unit_cell_limit),
            ignore_errors=False,
            overwrite=bool(args.overwrite),
            cpus=args.number_of_worker_processes,
            seed=args.dataseed
        )
        if int(args.pickle) == 2:
            dump(Dataset, open("db_pickle.pk", "wb"), compression=compression_alg)
            print("Pickle Dumped")
        if int(args.pickle) == 3:
            dump(Dataset, open("db_pickle.pk", "wb"), compression=compression_alg)
            print("Pickle Dumped")
            sys.exit()
    train_model(config, Dataset)

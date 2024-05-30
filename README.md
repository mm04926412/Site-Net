Repository used for generating results for [Deep InfoMax paper title here]. The repository is provided as is for the purposes of providing the code used to generate the results in the paper and for reproduction of the results.
## Environments

A conda environment .yaml file has been included which will allow the construction of an anaconda environment. This conda environment can be created using

##### conda env create -f sitenet_env.yaml

For this file to resolve properly the channel priority on conda must be set to strict. For convinience, create_sitenet_env.sh has been provided. This .sh file will store the current value of the channel priority, set the channel priority to strict, and then revert it back to whatever the previous setting was after the environment is installed. The default name of this environment is pytorch2201 but can be changed without consequence.

##### conda activate *Env Name*

This local environment contains the same versions of all key packages used to generate the original results if built using create_sitenet_env.sh

## Scripts for reproducing paper results

Step 1. Generate data

First generate the formation energy and band gap datasets with create_matbench_hdf5.py

### create_matbench_hdf5.py arguments

##### \--primitive generates a dataset of primitive unit cells

##### \--cubic_supercell generates a dataset of supercells

##### -s \--supercell_size allows the size of the supercells to be specified 

##### -w \--number_of_worker_processes allows the number of cpu threads used to be specified (default 1)

##### -d \--matbench_dataset determines the matbench dataset used for creating the hdf5 database

either \--primitive or \--cubic_supercell must be used

provide the size of the supercell (if applicable) with -s N where N is the maximum number of atoms.

After setting up the conda environment you can generate the formation energy and band gap dataset by running

python create_matbench_hdf5.py --cubic_supercell -s 50 -w [number of cpu cores desired] -d matbench_mp_gap
python create_matbench_hdf5.py --cubic_supercell -s 50 -w [number of cpu cores desired] -d matbench_mp_e_form

Step 2. Train Deep InfoMax models

Deep InfoMax models can be trained with DIM_train.py. This script trains a Deep InfoMax site-net with the hyper parameters in a yaml config file.

### DIM_train.py arguments

##### -c \--config allows the path of the configuration file to be specified (default None) 

##### -f \--h5_file_name allows the path of the h5 dataset used for training to be specified (default None) 

##### -l \--load_checkpoints allows training to be resumed from the most recent checkpoint for a given config file (default 0) 

##### -o \--overwrite will force the generation of new features, followed by overwriting, instead of reading them from the h5 file (default False) 

##### -n \--limit will limit the model to loading the first n samples, this is normally for debug purposes / test runs (default None) 

##### -u \--unit_cell_limit will exclude unit cells larger than this size from training (default 100) 

##### -w \--number_of_worker_processes controls the maximum number of cpu threads that site-net will use (default 1)

To reproduce the models used in this work run

python DIM_train.py -c compact_dim_nocomp_klnorm.yaml -f Data/Matbench/matbench_mp_e_form_cubic_50_train_1.hdf5 -u 50 -w [number of cpu cores]
python DIM_train.py -c compact_dim_nocomp_klnorm.yaml -f Data/Matbench/matbench_mp_gap_cubic_50_train_1.hdf5 -u 50 -w [number of cpu cores]

Tensorboard logs will be dumped to the lightning_logs folder, halt the model once the validation global DIM and local DIM loss scores have converged. Training can be resumed with -l 1 given the same arguments and configuration file.

Step 3. Train supervised site-net models using DIM starting paramaters

In this step the supervised site-nets can be trained. This is performed with DIM_train_downstream.py. On lines 87-94 dictionaries are defined that map the "names" of each kind of model and each dataset to the checkpoint / dataset paths. Please replace the paths with the names of the checkpoints and datasets in your repository.

### DIM_train_downstream.py arguments

##### -c \--config allows the path of the configuration file to be specified (default None)

##### -f \--h5_file_name allows the path of the h5 dataset used for training to be specified (default None) 

##### -l \--load_checkpoints allows training to be resumed from the most recent checkpoint for a given config file (default 0) 

##### -o \--overwrite will force the generation of new features, followed by overwriting, instead of reading them from the h5 file (default False) 

##### -n \--limit will limit the model to loading the first n samples, this is normally for debug purposes / test runs (default None) 

##### -u \--unit_cell_limit will exclude unit cells larger than this size from training (default 100) 

##### -w \--number_of_worker_processes controls the maximum number of cpu threads that site-net will use (default 1)

##### -s \--dataseed Allows the subset of the data chose for training the supervised model to be consistent, integer seeds are reccomended

#### -m --starting_model Allows the starting weights to be defined

#### -r --freeze Allows layers of the site-net to be frozen, in practice this never improved performance and is deprecated

To reproduce the models, please run the following set of scripts. every combination of parameters in the curly brackets must be run

python DIM_train_downstream.py -c {compact_dim_s50.yaml, compact_dim_s100.yaml, compact_dim_s250.yaml, compact_dim_s1000.yaml} -f eform -m {none, DIM_eform, Supervised_egap} -w num_cpus -u 50 -s {1,2,3,4,5,6,7,8,9,10,11,12}
python DIM_train_downstream.py -c {compact_dim_s50.yaml, compact_dim_s100.yaml, compact_dim_s250.yaml, compact_dim_s1000.yaml} -f egap -m {none, DIM_egap, Supervised_eform} -w num_cpus -u 50 -s {1,2,3,4,5,6,7,8,9,10,11,12}

This is quite time consuming, so a full suite of trained models have already been included in the Data/Matbench/downstream_models folder

Step 4. Get test MAEs on the downstream supervised site-nets and output to file

Train sklearn models and get test MAEs on the DIM derived representations with DIM_train_downstream.py. On line 56 the dictionary config and model contains the names of the models, the config for the hyper parameters of the model, the weights of the model, and the dataset to run on

### DIM_train_sklearn.py arguments

##### -u \--unit_cell_limit will exclude unit cells larger than this size from training (default 100) 

##### -w \--number_of_worker_processes controls the maximum number of cpu threads that site-net will use (default 1)

step 5. Run downstream sklearn models on Deep InfoMax representations and ouput test MAEs to file

Run predict_transfer_downstream.py to get the test MAEs for every supervised model checkpoints in the Data/Matbench/downstream_models folder. This will also implicitly produce the TSNE's for each model type

### predict_transfer_downstream.py arguments

##### -c \--config allows the path of the configuration file to be specified (default None)

##### -n \--limit will limit the model to loading the first n samples, this is normally for debug purposes / test runs (default None) 

##### -w \--number_of_worker_processes controls the maximum number of cpu threads that site-net will use (default 1)

##### -u \--unit_cell_limit will exclude unit cells larger than this size from training (default 100)

step 6. Run downstream sklearn models on traditional featurizers and output test MAES to file

Run Featurizer_Downstream_sklearn.py to generate the test MAEs for the sklearn models trained on top of the xray diffraction pattern and flattened orbital field matrix. Line 170 contains a dictionary pointing to the location of the band gap and formation energy train and test datasets, please modify the dictionary to point to the correct files.

### Featurizer_Downstream_sklearn.py arguments

##### -u \--unit_cell_limit will exclude unit cells larger than this size from training (default 100) 

##### -w \--number_of_worker_processes controls the maximum number of cpu threads that site-net will use (default 1)

##### -d \--dataset either egap or eform, determines whether to run on the band gap or formation energy dataset

#### -f \--featurizer Allows the featurizer to be specified, either orbital or xray

step 7. Generate performance plots for downstream supervised site-nets and representation learning in sklearn

Run through the notebook Downstream_MAEs_to_plots.ipynb to get the box plots from the paper

step 8. Generate the TSNEs from the paper

Run through the notebook TSNE_production.ipynb 

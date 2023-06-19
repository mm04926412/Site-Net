#!/bin/bash -l
# Use the current working directory
#SBATCH -D ./
# Use the current environment for this job
#SBATCH --export=ALL
# Define job name
#SBATCH -J major
# Define a standard output file. When the job is running, %N will be replaced by the name of 
# the first node where the job runs, %j will be replaced by job id number.
#SBATCH -o major.out
# Define a standard error file
#SBATCH -e major.err
# Request the GPU partition
#SBATCH -p gpu
# Request the number of GPUs to be used (if more than 1 GPU is required, change 1 into Ngpu, where Ngpu=2,3,4)
#SBATCH --gres=gpu:1
# Request the number of CPU cores. (There are 24 CPU cores and 4 GPUs on the GPU node,
# so please request 6*Ngpu CPU cores, i.e., 6 CPU cores for 1 GPU, 12 CPU cores for 2 GPUs, and so on.
# User may request more CPU cores for jobs that need very large CPU memory occasionally, 
# but the number of CPU cores should not be greater than 6*Ngpu+5. Please email hpc-support if you need to do so.)
#SBATCH -n 6
#SBATCH -N 1
# Set time limit in format a-bb:cc:dd, where a is days, b is hours, c is minutes, and d is seconds.
#SBATCH -t 3-00:00:00
# Request the memory on the node or request memory per core
# PLEASE don't set the memory option as we should use the default memory which is based on the number of cores 
# Insert your own username to get e-mail notifications (note: keep just one "#" before SBATCH)
#SBATCH --mail-user=sgmmora2@liverpool.ac.uk
# Notify user by email when certain event types occur
#SBATCH --mail-type=ALL
#
# Set your maximum stack size to unlimited
ulimit -s unlimited
# Set OpenMP thread number
export OMP_NUM_THREADS=1

# Load tensorflow and relevant modules
#module purge
#module load apps/anaconda3/5.2.0
#use source activate gpu to get the gpu virtual environment
conda activate pytorch_2201

# List all modules
module list

echo =========================================================
pwd
echo SLURM job: submitted  date = `date`      
date_start=`date +%s`

hostname

echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "GPU_DEVICE_ORDINAL   : $GPU_DEVICE_ORDINAL"

echo "Running GPU jobs:"

echo "Running GPU job hparam_search_supervised.py:"
python DIM_train_supervised.py -c config/compact_dim_Adam_supervised.yaml -f Data/Matbench/matbench_mp_gap_cubic_50_train_1.hdf5 -w $SLURM_NTASKS -u 50

date_end=`date +%s`
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo =========================================================   
echo SLURM job: finished   date = `date`   
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================   

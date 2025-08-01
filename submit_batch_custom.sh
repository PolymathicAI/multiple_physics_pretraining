#!/bin/bash 
#SBATCH --time=168:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="demo_multi_gpus"
#SBATCH --output=out/demo_multi_gpus.out
#SBATCH --error=err/demo_multi_gpus.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jc02423@surrey.ac.uk



export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=6

master_node=$SLURMD_NODENAME
VENVDIR=/venvs/
run_name="demo_multi_gpus"
config="basic_config"   # options are "basic_config" for all or swe_only/comp_only/incomp_only/swe_and_incomp
yaml_config="./config/mpp_avit_b_config.yaml"

module purge
module load GCC/12.3.0 
module load OpenMPI/4.1.5
module load Python/3.11.3
module load HDF5/1.14.0
module load CUDA/12.2.2


# conda activate pdebench
source mpp_env/bin/activate
set -x


# srun python `which torchrun` \
# 	--nnodes=$SLURM_JOB_NUM_NODES \
#  	--nproc_per_node=2 \
# 	--rdzv_id=$SLURM_JOB_ID \
# 		--rdzv_backend=c10d \
# 		--rdzv_endpoint=$master_node:29500 \
# 		train_basic.py --run_name $run_name --config $config --yaml_config $yaml_config --use_ddp

# # multi GPU training
# export CUDA_VISIBLE_DEVICES=0
# python train_basic.py --run_name $run_name --config $config --yaml_config $yaml_config

# Use torchrun for multi-GPU training:
srun python $(which torchrun) \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$master_node:29500 \
    train_basic.py --run_name $run_name --config $config --yaml_config $yaml_config --use_ddp
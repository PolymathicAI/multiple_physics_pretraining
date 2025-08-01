#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="inference"
#SBATCH --output=out/inference.out
#SBATCH --error=err/inference.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jc02423@surrey.ac.uk

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=6

module purge
module load GCC/12.3.0
module load OpenMPI/4.1.5
module load Python/3.11.3
module load HDF5/1.14.0
module load CUDA/12.2.2

source mpp_env/bin/activate
set -x

# Run inference (adjust rollout_steps/history_length as needed)
python inference_basic.py --yaml_config ./config/mpp_avit_b_config.yaml --config basic_config
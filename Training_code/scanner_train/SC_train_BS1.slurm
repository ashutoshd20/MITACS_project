#!/bin/bash
#SBATCH --partition=gpu-v100
#SBATCH	-c 4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node 1

module purge
module load cuda/11.3.0

source activate ashu
python SC_train_Ashu.py -en ASHU_encoder_BS1.h5 -fn_train /home/ashutosh.dwivedi/Code_NEW_PATH_SLURM_JOBS/train_pd_complete_adni_NEW_PATH.csv -fn_test /home/ashutosh.dwivedi/Code_NEW_PATH_SLURM_JOBS/test_pd_complete_adni_NEW_PATH.csv -batch_size 1
#!/bin/bash
#SBATCH --partition=gpu-v100
#SBATCH	-c 4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node 1

module purge
module load cuda/11.3.0

source activate ashu
python main_central_Ashu.py -fn_train /home/ashutosh.dwivedi/Code_NEW_PATH_SLURM_JOBS/train_pd_complete_adni_NEW_PATH.csv -fn_test /home/ashutosh.dwivedi/Code_NEW_PATH_SLURM_JOBS/test_pd_complete_adni_NEW_PATH.csv -batch_size 3 -alpha 1.0 -beta 1 -en mini_encoder_BS3.h5 -pd ASHU_PD_classifier_BS3.h5 -sc SC_classifier_BS3.h5
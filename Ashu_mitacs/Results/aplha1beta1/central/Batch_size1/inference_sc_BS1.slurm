#!/bin/bash
#SBATCH --partition=gpu-v100
#SBATCH	-c 4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node 1

module purge
module load cuda/11.3.0

source activate ashu
python inference_sc.py -fn /home/ashutosh.dwivedi/Code_NEW_PATH_SLURM_JOBS/test_pd_complete_adni_NEW_PATH.csv -en encoder_unlearned_BS1.h5 -sc SC_classifier_unlearned_BS1.h5 -o SC_BS1
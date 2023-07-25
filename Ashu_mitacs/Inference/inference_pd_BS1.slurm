#!/bin/bash
#SBATCH --partition=gpu-v100
#SBATCH	-c 4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node 1

module purge
module load cuda/11.3.0

source activate ashu
python inference_pd.py -fn /home/ashutosh.dwivedi/Code_NEW_PATH_SLURM_JOBS/test_pd_complete_adni_NEW_PATH.csv -en encoder_distributed_unlearned_BS3_17.h5 -pd classifier_PD_distributed_unlearned_BS3_17.h5 -o PD_BS3
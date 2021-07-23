#!/bin/bash
#SBATCH --job-name channelTrans
#SBATCH --array=51-80
#SBATCH --partition=suprib
#SBATCH --mem-per-cpu=1GB
#SBATCH --time=80:00:00
#SBATCH --error=/data/cees/hjyang3/PnP/test.err
#SBATCH --output=/data/cees/hjyang3/PnP/test.out
#
# Job steps:
cd $SLURM_SUBMIT_DIR
#
/data/cees/hjyang3/anaconda3/envs/fenicsproject/bin/python PnP_Case2.py $SLURM_ARRAY_TASK_ID
# end script

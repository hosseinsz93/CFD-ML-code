#!/bin/bash

## Slurm queue commands

#SBATCH -J CNNsi1          # job name
#SBATCH -o run.o%j          # output and error file name, %j is the job id
#SBATCH -p long-40core  # queue name
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -N 1      # total number of nodes
#SBATCH -t 0-48:00:00        # run time (hh:mm:ss)

module load pytorch-latest/1.13.0

srun python 01main.py --epochs 10000 

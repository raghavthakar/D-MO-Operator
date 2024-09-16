#!/bin/bash
#SBATCH --time=0-00:00:55
#SBATCH --constraint=skylake
#SBATCH --mem=16G
#SBATCH -c 8

module load conda

mkdir /nfs/stak/users/thakarr/hpc-share/hehe
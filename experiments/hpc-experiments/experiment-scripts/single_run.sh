#!/bin/bash
#SBATCH --time=0-12:00:00
#SBATCH --constraint=skylake
#SBATCH --mem=16G
#SBATCH -c 8

# module load conda

/home/thakarr/D-MO-Operator/build/MOD "/home/thakarr/D-MO-Operator/experiments/hpc-experiments/MOREP-2objs-easy.yaml" "/home/thakarr/D-MO-Operator/experiments/hpc-experiments/data/MOREP-2objs-easy" "mod"
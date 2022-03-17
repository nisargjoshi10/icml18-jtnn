#!/bin/bash
#SBATCH --job-name=jtvae
#SBATCH --account=pfaendtner
#SBATCH --partition=gpu-a40
#SBATCH --gpus=a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6 
#SBATCH --time=10:00:00
#SBATCH --mem=100G

python preprocess.py --train ../data/moses/train.txt --split 100 --jobs 16
mkdir moses-processed
mv tensor* moses-processed

#!/bin/bash
#SBATCH -J bert512
#SBATCH -A project_[project_number]
#SBATCH -t 16:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1,nvme:5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G
#SBATCH -o ../out/bert512.out

module purge
module load pytorch/1.7

echo 'Extracting data'
tar -xzf ../data.tar.gz -C $LOCAL_SCRATCH

export PYTHONUNBUFFERED=1

python train.py $LOCAL_SCRATCH/data/all ../models/bert512 $LOCAL_SCRATCH/data/topic_codes.txt

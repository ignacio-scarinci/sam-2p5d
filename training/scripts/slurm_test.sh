#!/bin/bash

#SBATCH --job-name=tiny_adpt_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --partition=multi
#SBATCH --output=../logs/%x-%j.out
#SBATCH --error=../logs/%x-%j.err
#SBATCH --time=2-00:00:00

. /etc/profile
module purge

ulimit -c unlimited
ulimit -s unlimited

source /home/iscarinci/.bashrc
source /home/iscarinci/miniconda3/bin/activate sam

srun python ../test.py trainer.experiment_name=tiny_adpt6  trainer.label_prompt=False trainer.point_pos=1 trainer.point_neg=1 trainer.bbox_prompt=False model.mod=sam_adpt 
#../main.py model=vit_b trainer.experiment_name=vit_b_adpt model.mod=sam_adpt

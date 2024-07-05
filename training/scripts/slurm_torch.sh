#!/bin/bash

#SBATCH --job-name=tiny
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --partition=multi
#SBATCH --output=../logs/%x-%j.out
#SBATCH --error=../logs/%x-%j.err
#SBATCH --time=2-00:00:00

. /etc/profile
module purge

ulimit -c unlimited
ulimit -s unlimited

source /home/iscarinci/.bashrc
micromamba activate sam

experiment_name=$SLURM_JOB_NAME

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

#export HYDRA_FULL_ERROR=1

srun torchrun \
--nnodes 2 \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29502 \
../main.py trainer.experiment_name=${experiment_name} model.mod=sam
#../main.py model=vit_b trainer.experiment_name=${experiment_name} model.mod=sam_adpt
#../main.py model=vit_l trainer.experiment_name=${experiment_name} model.mod=sam_adpt
#../main.py model=vit_h trainer.experiment_name=${experiment_name} model.mod=sam_adpt


mv ../logs/$SLURM_JOB_NAME-$SLURM_JOB_ID.* ../runs/${experiment_name}/

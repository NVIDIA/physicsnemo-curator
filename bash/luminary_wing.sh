#!/bin/bash
#SBATCH --job-name=prepro_wing
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=domino_%j.out
#SBATCH --error=domino_%j.err
#SBATCH --nodelist=inst-kfpue-gpu-prd

###SBATCH --nodelist=inst-t4s6p-gpu-prd

# Set distributed training environment variables
#export WORLD_SIZE=1
#export RANK=0
#export LOCAL_RANK=0
#export MASTER_ADDR=localhost
#export MASTER_PORT=29500

source /home/lserrano/physicsnemo/.venv/bin/activate

export PYTHONPATH=$PYTHONPATH:examples &&
physicsnemo-curator-etl --config-dir=examples/config --config-name=external_luminary_wing etl.source.input_dir=/nfs-gpu/research/datasets/luminary_wing/raw etl.sink.output_dir=/mnt/localdisk/luminary_wing/domino_format/full etl.common.model_type=combined



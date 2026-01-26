#!/bin/bash
#SBATCH --job-name=prepro_suv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=estate_%j.out
#SBATCH --error=estate_%j.err
##SBATCH --nodelist=inst-kfpue-gpu-prd

# Set distributed training environment variables
#export WORLD_SIZE=1
#export RANK=0
#export LOCAL_RANK=0
#export MASTER_ADDR=localhost
#export MASTER_PORT=29500


# Navigate to the script directory
#cd /home/lserrano/physicsnemo/examples/cfd/external_aerodynamics/domino/src

# Activate your environment
source /home/lserrano/physicsnemo/.venv/bin/activate

export HYDRA_FULL_ERROR=1

export PYTHONPATH=$PYTHONPATH:examples &&
physicsnemo-curator-etl --config-dir=examples/config --config-name=external_luminary_suv etl.source.input_dir=/nfs-gpu/research/datasets/luminary_suv/raw/full/AeroSUV_full_scale_estate_transient etl.sink.output_dir=/nfs-gpu/research/datasets/luminary_suv/domino_format/full/AeroSUV_full_scale_estate_transient etl.common.model_type=combined
#physicsnemo-curator-etl --config-dir=examples/config --config-name=external_luminary_suv etl.source.input_dir=/nfs-gpu/research/datasets/luminary_suv/raw/full/AeroSUV_full_scale_estate_transient etl.sink.output_dir=/nfs-gpu/research/datasets/luminary_suv/domino_format/full/AeroSUV_full_scale_estate_transient etl.common.model_type=combined



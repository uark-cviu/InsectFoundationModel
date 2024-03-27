#!/bin/bash
set -e
set -x

# Parameters
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --job-name=diffusion
#SBATCH --partition=gpu72
#SBATCH --time=3-00:00:00
#SBATCH --nodelist=c[1612-1615]
#SBATCH --qos=gpu

echo "NODELIST="${SLURM_NODELIST}

echo "MASTER_ADDR="$MASTER_ADDR
echo $SLURM_PROCID

source ~/virtualenv/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}

DATA_PATH=data/images_list.txt
model=vit_base_patch16
ckpt_dir=ckpt/insect_foundation_${model}
log_dir=${ckpt_dir}/log

export OFFSET_RANK=${1}
srun python pretrain.py \
    --batch_size 256 \
    --model ${model} \
    --norm_pix_loss \
    --mask_ratio 0.5 \
    --epochs 200 \
    --warmup_epochs 5 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --num_workers 8 \
    --start_epoch 0 \
    --data_path ${DATA_PATH} \
    --output_dir ${ckpt_dir} \
    --log_dir ${log_dir}
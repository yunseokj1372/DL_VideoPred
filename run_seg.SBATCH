#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu
#SBATCH --job-name=dl_video_1

module purge

singularity exec --nv \
	    --overlay /scratch/yj2369/dlenv/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif\
	    /bin/bash -c "source /ext3/env.sh; python /scratch/yj2369/DL_VideoPred/segmentation/seg_run.py"

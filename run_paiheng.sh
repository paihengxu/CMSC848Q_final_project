#!/bin/bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --partition=gpu
#SBATCH --time=04:00:00                                         # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --qos=gpu-medium                                            # set QOS, this will determine what resources can be requested
#SBATCH --mem=32g
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --exclude=materialgpu00,clipgpu04,clipgpu05,clipgpu06

source /cliphomes/paiheng/.bashrc
conda activate 848_hw1

data_fn=${1}
if_biased=${2}

python Q_Pain.py --medical_context_file ${data_fn} --closed_prompt ${if_biased}

conda deactivate
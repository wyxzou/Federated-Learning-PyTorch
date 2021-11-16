#!/bin/sh
#SBATCH --account=def-desterck
#SBATCH	--nodes=1
#SBATCH	--gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:20:00

export SLURM_TMPDIR=/home/wyzou/Federated-Learning-PyTorch/
module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch torchvision tqdm tensorboardX six --no-index

srun ./repeat_experiment.sh

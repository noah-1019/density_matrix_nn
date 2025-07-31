#!/bin/bash
#SBATCH --job-name=network_entropy              # the name of your job
#SBATCH --output=/scratch/njp257/network_entropy/outputs/output.txt      # this is the file your output and errors go to
#SBATCH --time=60                          # 1 hour
#SBATCH --chdir=/scratch/njp257/network_entropy                  # your work directory
#SBATCH --mem=32G                              # 32GB of memory
#SBATCH --mail-type=FAIL,END
#SBATCH --cpus-per-task=12   # or however many you want

# Run your application: precede the application command with 'srun'
# A couple example applications...

srun date

echo "SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE"

srun python3 generate_data.py

srun date
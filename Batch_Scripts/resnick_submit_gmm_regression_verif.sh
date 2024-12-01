#!/bin/bash
#### resnick_submit_gmm_regression_verif.sh START ####
#SBATCH --time=072:00:00                   # walltime
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G                          # memory per CPU core
#SBATCH -J "gmm_regression_verification"   # job name
#SBATCH --mail-user=glavrent@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#variable definition
export FREQ=5.011872 #frequency
export FLAG_PROD=0   #production run
export VERIF_RLZ=1   #random realization number
export FLAG_HETERO=1 #
export FLAG_UPD_GS=1 #

# echo job info on joblog:
echo "Job $SLURM_JOB_ID running: $SLURM_JOB_NAME"
echo "Job $SLURM_JOB_ID started on: " `hostname -s`
echo "Job $SLURM_JOB_ID started on: " `date `
echo "GMM Regression (Verification)"
echo "Frequency $FREQ hz"
echo "Heteroscedasticity: $FLAG_HETERO"
echo "Updated geometrical spreading: $FLAG_UPD_GS"
echo " "

# load the job environment:
module load gcc
module load tbb
module load openssl
source ~/miniconda3/etc/profile.d/conda.sh

# run commands
conda activate /central/groups/enceladus/NGAWest3_GMMdev/Analyses/conda_env_gmmdev
cd /home/glavrent/enceladus/NGAWest3_GMMdev/Analyses/gmm_ergodic/regression/
# regression
python regression_erg_gmm_regionalized.py

# echo job info on joblog:
echo "Job $SLURM_JOB_ID running: $SLURM_JOB_NAME"
echo "Job $SLURM_JOB_ID started on: " `hostname -s`
echo "Job $SLURM_JOB_ID started on: " `date `
echo "GMM Regression (Verification)"
echo "Frequency $FREQ hz"
echo "Heteroscedasticity: $FLAG_HETERO"
echo "Updated geometrical spreading: $FLAG_UPD_GS"
echo " "
#### resnick_submit_gmm_regression_verif.sh STOP ####

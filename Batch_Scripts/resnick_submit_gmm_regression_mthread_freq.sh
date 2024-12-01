#!/bin/bash
#### resnick_submit_gmm_regression_freq_single_mthread.sh START ####
#SBATCH --time=072:00:00                   # walltime
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=34G                          # memory 
#SBATCH -J "gmm_reg_freq_single_mthread"   # job name
#SBATCH --mail-user=glavrent@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#variable definition
#export FREQ=0.2511886 #frequency
#export FREQ=0.5011872 #frequency
export FREQ=1.000000  #frequency
#export FREQ=5.011872  #frequency
export FLAG_PROD=1    #production run
export FLAG_HETERO=1  #heteroscedasticity
export FLAG_UPD_ST=1  #updated geometrical spreading (short distance saturation)
export FLAG_HW=1      #include haningwall scaling
export FLAG_NL=1      #include non-linear site scaling

#MCMC options
#number of markov chains
export N_CHAINS=4
#multi-threading options
export FLAG_MTHREAD=1
export THREADS_CHAIN=6

# echo job info on joblog:
echo "Running GMM Regression (Production)"
echo "Job $SLURM_JOB_ID running: $SLURM_JOB_NAME"
echo "Job $SLURM_JOB_ID started on: " `hostname -s`
echo "Job $SLURM_JOB_ID started on: " `date `
echo "Frequency $FREQ hz"
echo "Heteroscedasticity: $FLAG_HETERO"
echo "Updated short-distance saturation: $FLAG_UPD_ST"
echo " "

# load the job environment:
module load boost
module load gcc
module load tbb
module load openssl
source ~/miniconda3/etc/profile.d/conda.sh

#activate conda environment
if  [[ "$FLAG_MTHREAD" -eq 1 ]]; then
  echo "multitreading implementation"
  conda activate /central/groups/enceladus/NGAWest3_GMMdev/Analyses/conda_env_gmmdev-mt
else
  conda activate /central/groups/enceladus/NGAWest3_GMMdev/Analyses/conda_env_gmmdev
fi
# run commands
cd /home/glavrent/enceladus/NGAWest3_GMMdev/Analyses/gmm_ergodic/regression/
# regression
python regression_erg_gmm_regionalized.py

# echo job info on joblog:
echo "Completed GMM Regression (Production)"
echo "Job $SLURM_JOB_ID completed: $SLURM_JOB_NAME"
echo "Job $SLURM_JOB_ID started on: " `hostname -s`
echo "Job $SLURM_JOB_ID started on: " `date `
echo "Frequency $FREQ hz"
echo "Heteroscedasticity: $FLAG_HETERO"
echo "Updated short-distance saturation: $FLAG_UPD_ST"
echo " "
#### resnick_submit_gmm_regression_freq_single_mthread.sh STOP ####

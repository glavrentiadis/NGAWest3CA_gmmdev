#!/bin/bash
#### resnick_submit_gmm_regression_freq_single_mthread.sh START ####
#SBATCH --time=072:00:00                   # walltime
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=34G                          # memory 
#SBATCH --array=0-41                       # batch array
#SBATCH -J "gmm_reg_freq_single_mthread"   # job name
#SBATCH --mail-user=glavrent@caltech.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#frequency array
FREQ_ARRAY=(0.01318257        0.01513562          0.019952621         0.02511887          0.03019952 0.03981072   0.05011873 0.06025596 0.075857751 0.079432822           0.095499262 \
            0.10000           0.15135611          0.1995262           0.2511886           0.3019952  0.39810714   0.5011872  0.6025595  0.7585776 \
            1.00000 1.258926  1.513561            1.9952621           2.5118863           3.019952   3.981071     5.011872   6.025596   7.585776                 8.5113792 \
            10.0000 12.022642 15.135614  16.98244 19.952621 21.877611 25.11886  27.542291 30.19952   35.481334    39.81071)

#variable definition
export I_FREQ=$SLURM_ARRAY_TASK_ID #frequency index
export FREQ=${FREQ_ARRAY[$I_FREQ]} #frequency
export FLAG_PROD=1   		   #production run
export FLAG_HETERO=1 		   #heteroscedasticity
export FLAG_UPD_ST=1 		   #updated geometrical spreading (short distance saturation)
export FLAG_HW=1     		   #include haningwall scaling
export FLAG_NL=1     		   #include non-linear site scaling

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
echo "Max Task: $SLURM_ARRAY_TASK_MIN"
echo "Min Task: $SLURM_ARRAY_TASK_MAX"
echo "Current Task: $SLURM_ARRAY_TASK_ID"
echo "Frequency $FREQ hz"
echo "Heteroscedasticity: $FLAG_HETERO"
echo "Updated short-distance saturation: $FLAG_UPD_ST"
echo " "

pause time
#pause_time=$((60*2*SLURM_ARRAY_TASK_ID))
pause_time=$((4*SLURM_ARRAY_TASK_ID))
sleep ${pause_time}s

# load the job environment:
module load boost
module load gcc
module load tbb
module load openssl
source ~/miniconda3/etc/profile.d/conda.sh

#activate conda environment
if  [[ "$FLAG_MTHREAD" -eq 1 ]]; then
  conda activate /central/groups/enceladus/NGAWest3_GMMdev/Analyses/conda_env_gmmdev-mt
else
  conda activate /central/groups/enceladus/NGAWest3_GMMdev/Analyses/conda_env_gmmdev
fi
# run commands
cd /home/glavrent/enceladus/NGAWest3_GMMdev/Analyses/gmm_ergodic/regression/
# regression
stdbuf -oL python regression_erg_gmm_regionalized.py

# echo job info on joblog:
echo "Completed GMM Regression (Production)"
echo "Job $SLURM_JOB_ID completed: $SLURM_JOB_NAME"
echo "Job $SLURM_JOB_ID started on: " `hostname -s`
echo "Job $SLURM_JOB_ID started on: " `date `
echo "Max Task: $SLURM_ARRAY_TASK_MIN"
echo "Min Task: $SLURM_ARRAY_TASK_MAX"
echo "Current Task: $SLURM_ARRAY_TASK_ID"
echo "Frequency $FREQ hz"
echo "Heteroscedasticity: $FLAG_HETERO"
echo "Updated short-distance saturation: $FLAG_UPD_ST"
echo " "
#### resnick_submit_gmm_regression_freq_single_mthread.sh STOP ####

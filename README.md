# NGA3CA Ground Motion Model Development <br> (Dr. Abrahamson's group)
This repository contains the files used in the development of Abrahamson's group ergodic and non-ergodic ground motion models as part of the Next Generation Attenuation 3 project.


## Instructions
### Installation of Jupyter Python and R  Kernel 
Note the following instructions require that you have access to the shared Desing Safe Project (PRJ-4291)

#### Jupyter Python Kernel
To install the shared ipython kernel, which includes all shared libraries
* Start Jupyter Lab from Design-Safe
* Lunch the Terminal from the Other subcategory
* Activate shared conda environment: `conda activate /home/jupyter/projects/PRJ-4291/conda_env`
* Install the IPython kernel: <br> `/home/jupyter/projects/PRJ-4291/conda_env/bin/ipython kernel install --user --name=gmm_env_python`
* Restart Jupyter Lab on Desing-Safe; the python_gmm_env_shared kernel should appear in the list of available kernels.
#### Jupyter R Kernel
To install the shared R kernel, which includes all common packages for GMM development (e.g., lmer, ggplot, tidyverse)
* Start Jupyter Lab from Design-Safe
* Lunch the Terminal from the Other subcategory
* Activate shared conda environment: `conda activate /home/jupyter/projects/PRJ-4291/conda_env`
* Lunch R: `R`
* Install IRkernel: `IRkernel::installspec(name='gmm_env_r', displayname='gmm_env_R')`
* Restart Jupyter Lab on Desing-Safe; the R_gmm_env_shared kernel should appear in the list of available kernels.

## Troubleshooting
 

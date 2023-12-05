# NGA West 3 CA Ground Motion Model Development <br> (Dr. Abrahamson's group)
This repository contains the files used in the development of Abrahamson's group ergodic and non-ergodic ground motion models as part of the Next Generation Attenuation West 3 project.

## Table of Contents
1. [Repository Structure](#repository-structure)
2. [Instructions](#instructions)
3. [Troubleshooting](#troubleshooting)

## Repository Structure
    .
    ├── Analyses                   (regression scripts)
    │    ├── gmm_ergodic           (scripts related to the ergodic gmm development)
    │    │    ├── preprocessing    (preprocessing of raw input files)
    |    |    ├── regression
    |    |    └── comparisons
    |    ├── gmm_nonergodic
    │    │    ├── preprocessing
    |    |    ├── regression
    |    |    └── comparisons
    |    ├── julia_lib
    |    ├── matlab_lib
    |    ├── python_lib
    |    └── r_lib
    |
    ├── Data                       (output files)
    │    ├── gmm_ergodic
    |
    ├── Raw_files                  (input files in original format)
    ├── Meetings                   (group meeting material)
    ├── Reporting                  (publications' folder)
    ├── conda_env                  (shared python/r conda environment for gmm development)
    ├── .git 
    |
    ├── README.md 
    ├── LICENSE 
    └── .gitignore                       

## Instructions
### Installation of Jupyter Python and R  Kernel 
Note the following instructions require that you have access to the shared Desing Safe Project (PRJ-4291)

#### Jupyter Python Kernel
To install the shared ipython kernel, which includes all shared libraries
* Start Jupyter Lab from Design-Safe
* Lunch the Terminal from the Other subcategory
* Activate shared conda environment: `conda activate /home/jupyter/projects/PRJ-4291/conda_env`
* Install the IPython kernel: <br> `/home/jupyter/projects/PRJ-4291/conda_env/bin/ipython kernel install --user --name=gmm_env_python`
* Exit Terminal: `exit`
* Restart Jupyter Lab on Desing-Safe; the python_gmm_env_shared kernel should appear in the list of available kernels.
#### Jupyter R Kernel
To install the shared R kernel, which includes all common packages for GMM development (e.g., lmer, ggplot, tidyverse)
* Start Jupyter Lab from Design-Safe
* Lunch the Terminal from the Other subcategory
* Activate shared conda environment: `conda activate /home/jupyter/projects/PRJ-4291/conda_env`
* Lunch R: `R`
* Install IRkernel: `IRkernel::installspec(name='gmm_env_r', displayname='gmm_env_R')`
* Exit R: `q()`
* Exit Terminal: `exit`
* Restart Jupyter Lab on Desing-Safe; the R_gmm_env_shared kernel should appear in the list of available kernels.

# Troubleshooting
The inventory below contains a list of commonly encountered issues organized by the type of problem.

##### Missing packages or Syntax error
 * Ensure the appropriate Kernel has been selected
   * Go to `Kernel` on the menu bar,
   * Select `Change Kernel ...` from the drop-down list
   * Select `gmm_env_R` or `gmm_env_python` on the menu for a R or Python Jupyter notebook, respectively 

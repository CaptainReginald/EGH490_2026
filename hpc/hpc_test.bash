#!/bin/bash

#PBS -N test_job
#PBS -l walltime=00:10:00
#PBS -l ncpus=1
#PBS -l mem=1gb
#PBS -m abe

cd $PBS_O_WORKDIR

source /home/wardlewo/miniforge3/bin/activate cgras

python3 /home/wardlewo/hpc-home/Corals/cgras_settler_counter/hpc/test.py

conda deactivate
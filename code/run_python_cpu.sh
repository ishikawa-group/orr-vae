#!/bin/bash
#$ -cwd
#$ -l cpu_80=1
#$ -l h_rt=24:00:00
#$ -N RPBE_ORR
#$ -o /gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/result/log/RPBE_output.log
#$ -e /gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/result/log/RPBE_error.log

# Load required modules
module load intel
module load intel-mpi
module load cuda

# Activate Python virtual environment
source /gs/fs/tga-ishikawalab/wakamiya/python_virtual_env/ORR_catalyst_generator_env/bin/activate

# Set VASP pseudopotential path for ASE
export VASP_PP_PATH=/gs/fs/tga-ishikawalab/vasp/potential
export VASP_SCRIPT=/gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/run_vasp/run_vasp.py

# Run python script
python3 /gs/fs/tga-ishikawalab/wakamiya/ORR_catalyst_generator/code/run_test.py
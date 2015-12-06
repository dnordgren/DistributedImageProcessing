#!/bin/bash
# call this script with the following parameters:
#     input file
#     number of cores to run the job with

# build the slurm file
echo "#SBATCH --ntasks=$1"               > job.slurm
echo "#SBATCH --time=03:15:00"          >> job.slurm
echo "#SBATCH --job-name=jib-job"       >> job.slurm
echo "#SBATCH --error=err.err"          >> job.slurm
echo "#SBATCH --output=out.out"         >> job.slurm
echo "mpiexec -n $1 main.out $2 $3 $4"  >> job.slurm

# queue the slurm job
sbatch job.slurm

exit 0


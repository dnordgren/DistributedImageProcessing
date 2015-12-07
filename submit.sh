#!/bin/bash
# call this script with the following parameters:
#     number of cores to run the job with
#     input file
#     number of chunks to divide image into
#     operation type (0 or 1)

# build the slurm file
echo "#!/bin/sh"                         > job.slurm
echo "#SBATCH --ntasks=$1"              >> job.slurm
echo "#SBATCH --time=03:15:00"          >> job.slurm
echo "#SBATCH --job-name=jib-job"       >> job.slurm
echo "#SBATCH --error=err.out"          >> job.slurm
echo "#SBATCH --output=out.out"         >> job.slurm
echo "mpiexec -n $1 main.out $2 $3 $4"  >> job.slurm

# cleanup from last run
rm *png

# queue the slurm job
sbatch job.slurm

exit 0


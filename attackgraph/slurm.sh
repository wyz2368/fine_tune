#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=attacks
#SBATCH --mail-user=yongzhao_wang@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=7g
#SBATCH --time=300:00:00
#SBATCH --account=wellman
#SBATCH --partition=standard

# The application(s) to execute along with its input arguments and options:

module load python3.6-anaconda/5.2.0
cd ${SLURM_SUBMIT_DIR}
python deepgraph_runner.py > out.txt

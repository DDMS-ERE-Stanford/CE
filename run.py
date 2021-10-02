###########################################################
## write and run slurm batch file #########################
###########################################################

import os 
import sys 
import random
from util.misc import *

if __name__ == '__main__': 
    input_dir = sys.argv[1] + "/"
    check_input_dir = os.path.isdir( input_dir ) 
    if not check_input_dir: 
        sys.exit("input case folder need to exist")
    input_file = sys.argv[2]
    simul_params, channel_params, obs_params, pnp_params = readInput( input_file, input_dir )

    ce_type = simul_params["ce_type"] 
    if ce_type == 0:
        job_type = "Deterministic" 
    elif ce_type == 1: 
        job_type = "Probabilistic"
        samples = random.sample(range(1,channel_params["total_samples"]), channel_params["num_ens"]) 
        
    job_name = "CE_" + job_type 
    job_file = os.getcwd() + "/run_" + job_type + ".sh"
    python_path = "/data/cees/hjyang3/anaconda3/envs/fenicsproject/bin/python"
    src = job_type + ".py"

    with open(job_file, "w" ) as f: 
        f.writelines("#!/bin/bash\n")
        f.writelines("#SBATCH --job-name %s\n" % job_name)
        if ce_type == 1:
            f.write("#SBATCH --array=")
            N = len(samples)
            for ind in range(N-1): 
                f.write( str(samples[ind]) + ",")
            f.write( str(samples[N-1]) + "\n")
        f.writelines( "#SBATCH --partition=suprib\n" ) 
        f.writelines( "#SBATCH --mem-per-cpu=1GB\n" ) 
        f.writelines( "#SBATCH --time=80:00:00\n" )
        f.writelines( "#SBATCH --error=%s/%s.err\n" % (os.getcwd(), job_name) )
        f.writelines( "#SBATCH --output=%s/%s.out\n" % (os.getcwd(), job_name) )
        f.writelines( "#\n" )
        f.writelines( "# Job steps:\n" ) 
        f.writelines( "cd $SLURM_SUBMIT_DIR\n")        
        f.writelines( "#\n" )
        if ce_type == 0: 
            f.writelines( "%s %s %s %s\n" % (python_path, src, input_dir, input_file) )
        elif ce_type == 1: 
            f.writelines( "%s %s $SLURM_ARRAY_TASK_ID %s %s\n" % (python_path, src, input_dir, input_file) )
        f.writelines( "# end script" )

    os.system("sbatch %s" %job_file) 
    

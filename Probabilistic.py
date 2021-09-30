###########################################################
## CE_probabilistic.py
## Main for probablistic inversion using CE algorithm 
##########################################################

import numpy as np 
import dolfin as df 
import sys 
import os 
from time import time 
from util.misc import *
from util.plot import * 
from util.sampling import * 
from util.forward import *
from model.initial import *
from CE_algorithm import * 

# simulation of main CE algorithm    
def probabilistic_main( ind, input_dir, input_file ): 
    start_time = time()
    data_dir, log_dir, output_dir = get_case_dirs( input_dir ) 
    if "Case1" in input_dir:
        vae_case = 1 
    elif "Case2" in input_dir: 
        vae_case = 2
    ref_file = channel_params["true_file"]
    mps_file = channel_params["mps_file"] 
    simul_params, channel_params, obs_params, pnp_params = readInput( input_file, input_dir )
    m_true, d_obs = getChannelTrue( channel_params, simul_params, ref_file, data_dir)
    log_file = "log_" + str(ind) + ".txt" 
    _, obs_vec = obsMatrix( obs_params["nxblock_dyn"], obs_params["nyblock_dyn"], simul_params["ngx"], simul_params["ngy"] ) 
    f_log = open( log_dir + log_file, "w" ) 
    m_ini = getMPSInitial( ind, channel_params, simul_params, mps_file, data_dir ) 
    m_1 = m_ini
    m_2 = m_ini
    m_3 = m_ini
    m_new = m_ini
    ce_iter = 1
    while True: 
        m_old = m_new
        f_log.write( "Iteration: %d \n" %ce_iter )
        if (pnp_params["num_models"] == 2): 
            m_1, m_2 = one_step_dat_den( m_1, m_2, d_obs, obs_vec, channel_params, simul_params, pnp_params ) 
            m_new = ( m_1 + m_2 )/2
        elif (pnp_params["num_models"] == 3): 
            m_1, m_2, m_3 = one_step_dat_den_vae( m_1, m_2, m_3, 
                                         d_obs, obs_vec, vae_case, channel_params, simul_params, pnp_params ) 
            m_new = ( m_1 + m_2 + m_3 )/3
        res = getResidual( m_old, m_new )
        obj, d_new = getDataLoss( d_obs, m_new, obs_vec, channel_params, simul_params ) 
        f_log.write("Data misfit: %6.2f, residual: %2.6f \n"%(obj,res))
        if ( res < pnp_params["tol"] or ce_iter > pnp_params["max_iter"] ): 
           break
        ce_iter += 1  
    f_log.write("Ensemble number %d, time : %5.2f s \n" %(ind, time()-start_time))
    f_log.close()
    if (pnp_params["num_models"] == 2): 
        output_file_path = output_dir + "updated_field_data_den_" + str(ind) + ".txt"
    elif (pnp_params["num_models"] == 3): 
        output_file_path = output_dir + "updated_field_data_den_vae_" + str(ind) + ".txt"
    np.savetxt( output_file_path, m_new )
 

if __name__ == '__main__': 
    ind = sys.argv[1]
    ind = int(ind)
    input_dir = sys.argv[2]
    input_file = sys.argv[3]
    probabilistic_main( ind, input_dir, input_file )

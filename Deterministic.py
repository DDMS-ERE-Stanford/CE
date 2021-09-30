############################################################
## Deterministic.py
## Main algorithm for deterministic inversion using CE algorithm 
############################################################

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

##  main deterministic CE algorithm
def deterministic_main( input_dir, input_file ): 
    start_time = time()
    data_dir, log_dir, output_dir = get_case_dirs( input_dir ) 
    simul_params, channel_params, obs_params, pnp_params = readInput( input_file, input_dir )
    m_true, d_obs = getChannelTrue( channel_params, simul_params, channel_params["true_file"], data_dir)
    log_file = "log_" + str(ind) + ".txt" 
    _, obs_vec = obsMatrix( obs_params["nxblock_dyn"], obs_params["nyblock_dyn"], simul_params["ngx"], simul_params["ngy"] ) 
    f_log = open( log_dir + log_file, "w" ) 
    if (channel_params["initial"] == "Gauss"): 
        m_ini = getGaussInitial( m_true, channel_params, simul_params, obs_params )
    elif (channel_params["initial"] == "SVR"):
        m_ini = getSVMInitial( m_true, channel_params, simul_params, obs_params )
    m_1 = m_ini
    m_2 = m_ini
    m_new = m_ini
    ce_iter = 1
    while True: 
        m_old = m_new
        f_log.write( "Iteration: %d \n" %ce_iter )
        m_1, m_2 = one_step_dat_den( m_1, m_2, d_obs, obs_vec, channel_params, simul_params, pnp_params ) 
        m_new = ( m_1 + m_2 )/2
        res = getResidual( m_old, m_new )
        obj, d_new = getDataLoss( d_obs, m_new, obs_vec, channel_params, simul_params ) 
        f_log.write("Data misfit: %6.2f, residual: %2.6f \n"%(obj,res))
        if ( res < pnp_params["tol"] or ce_iter > pnp_params["max_iter"] ): 
           break
        ce_iter += 1  
    f_log.write("Denoiser: %s, time : %5.2f s \n" %(pnp_params["denoiser"], time()-start_time))
    f_log.close()
    np.savetxt( output_dir + pnp_params["denoiser"] + ".txt", m_new )  
    plotField( m, simul_params, "perm_output_" + pnp_params["denoiser"], output_dir) 

 
if __name__ == '__main__': 
    input_dir = sys.argv[1]
    input_file = sys.argv[2]
    deterministic_main( input_dir, input_file )

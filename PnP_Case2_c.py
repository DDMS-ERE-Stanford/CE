import numpy as np 
import dolfin as df 
import sys 
import os 
from time import time 
from util.misc import *
from util.plot import * 
from util.sampling import * 
from util.forward import *
from model.posterior import *
from model.denoisePrior import * 
from model.initial import *
from model.dncnnPrior import * 
from model.vaePrior4 import *
import shutil
import multiprocessing 
from functools import partial 

home_dir = "/data/cees/hjyang3/PnP/"
folder = "Case5/"
input_dir = home_dir + folder 
data_dir = home_dir + folder + "data/"
output_dir = home_dir + folder + "output/" 
log_dir = home_dir + folder + "log/"
input_file = "Input.txt" 
Ref_file = "Ref.txt"
MPS_file = "Train.txt" 
simul_params, channel_params, obs_params, pnp_params = readInput( input_file, input_dir )

def getChannelTrue( channel_params, simul_params, input_file, input_dir):
    m_true,_ = getInput(input_dir, input_file, 1)
    m_true *= (channel_params["hperm"] - channel_params["lperm"]) 
    m_true += channel_params["lperm"] 
    d_true = forwardWellTrans( m_true, simul_params)
    print(d_true.shape) 
    return m_true, d_true

def dataMisfit( d, d_obs, obs_vec, simul_params): 
    nstep = simul_params["num_steps"]
    obj = 0
    for n in range(nstep):
        d_ = d[:, n+1:n+2]
        d_obs_ = d_obs[:, n+1:n+2]
        obj += 0.5 * np.dot((obs_vec*(d_-d_obs_)).T, obs_vec*(d_-d_obs_))
    obj /= nstep
    return obj   

def printDataLoss( d_obs, m, obs_vec, simul_params): 
    d = forwardWellTrans( m, simul_params)
    obj = dataMisfit( d, d_obs, obs_vec, simul_params )
    return obj, d 
  
def pnpResidual(m_old, m_new):
    N = m_old.shape[0] 
    res = np.linalg.norm(m_old - m_new) / N
    return res

def saveOutput( m_true, m_initial_final, m_new_final, d_obs, d_initial_final, d_new_final, simul_params, output_dir ):
    np.savetxt( output_dir + 'initials_field.txt', m_initial_final ) 
    np.savetxt( output_dir + 'updated_field.txt', m_new_final )  
    m_initial_avg = np.mean( m_initial_final, axis = 1, keepdims = True ) 
    m_new_avg = np.mean( m_new_final, axis = 1, keepdims = True ) 
    d_new_avg = np.mean( d_new_final, axis = 2, keepdims = False ) 
    d_initial_avg = np.mean( d_initial_final, axis = 2, keepdims = False)
    plotField( m_initial_avg, simul_params, "initial_field_avg", output_dir )
    plotField( m_new_avg, simul_params, "updated_field_avg", output_dir )
    plotField( m_true, simul_params, "true_field", output_dir ) 
    plotFieldTstep( d_obs, simul_params, "true_head", output_dir )    
    plotFieldTstep( d_new_avg, simul_params, "update_head_avg", output_dir ) 
    plotFieldTstep( d_initial_avg, simul_params, "initial_head_avg", output_dir )
 
def PnP_p_d( ind  ): 
    start_time = time()
    m_true, d_obs = getChannelTrue( channel_params, simul_params, Ref_file, data_dir)
    log_file = "log_c_" + str(ind) + ".txt" 
    _, obs_vec = obsMatrix( obs_params["nxblock_dyn"], obs_params["nyblock_dyn"], simul_params["ngx"], simul_params["ngy"] ) 
    f = open( log_dir + log_file, "w" ) 
    # read initials 
    if (channel_params["initial"] == "Gauss"):
        m_ini = gaussInitial( m_true, channel_params, simul_params, obs_params ) 
    elif (channel_params["initial"] == "SVR"): 
        m_ini = svmInitial( m_true, channel_params, simul_params, obs_params ) 
    else: 
        m_ini = getMPSInitial( ind, channel_params, simul_params, MPS_file, data_dir ) 
    m_1 = m_ini
    m_2 = m_ini
    m_3 = m_ini
    m_new = m_ini
    q = 1
    obj, d_ini = printDataLoss( d_obs, m_ini, obs_vec, simul_params ) 
    while True: 
        m_old = m_new
        f.write( "Iteration: %d \n" %q )
        m_1_new = 2 * posteriorTransProx( d_obs, m_1, m_1, obs_vec, channel_params, simul_params, pnp_params ) - m_1 
        if ( pnp_params["denoiser"] == "bm3d"): 
            m_2_new = bm3dPrior(m_2, simul_params, pnp_params)
        elif (pnp_params["denoiser"] == "TV"): 
            m_2_new = tvPrior(m_2, simul_params)
        elif (pnp_params["denoiser"] == "dncnn"):
            m_2_new = dncnnPrior(m_2, channel_params, simul_params)
        m_2_new = 2 * m_2_new - m_2
        # 2G - I 
        m_1 = (1 - pnp_params["rho"]) * m_1 + pnp_params["rho"] *  m_2_new  
        m_2 = (1 - pnp_params["rho"]) * m_2 + pnp_params["rho"] *  m_1_new
        m_new = ( m_1 + m_2 )/2
        res = pnpResidual( m_old, m_new )
        obj, d_new = printDataLoss( d_obs, m_new, obs_vec, simul_params ) 
        f.write("Data misfit: %6.2f, residual: %2.6f \n"%(obj,res))
        if ( res < pnp_params["tol"] or q > pnp_params["max_iter"] ): 
           break
        q += 1  
    f.write("Ensemble number %d, time : %5.2f s \n" %(ind, time()-start_time))
    f.close()
    np.savetxt( output_dir + "m_c_" + str(ind) + ".txt", m_new)
 
def PnP_p_d_v( ind  ): 
    start_time = time()
    m_true, d_obs = getChannelTrue( channel_params, simul_params, Ref_file, data_dir)
    log_file = "log_" + str(ind) + ".txt" 
    _, obs_vec = obsMatrix( obs_params["nxblock_dyn"], obs_params["nyblock_dyn"], simul_params["ngx"], simul_params["ngy"] ) 
    f = open( log_dir + log_file, "w" ) 
    # read initials 
    if (channel_params["initial"] == "Gauss"):
        m_ini = gaussInitial( m_true, channel_params, simul_params, obs_params ) 
    elif (channel_params["initial"] == "SVR"): 
        m_ini = svmInitial( m_ture, channel_params, simul_params, obs_params ) 
    else: 
        m_ini = getMPSInitial( ind, channel_params, simul_params, MPS_file, data_dir ) 
    m_1 = m_ini
    m_2 = m_ini
    m_3 = m_ini
    m_new = m_ini
    q = 1
    obj, d_ini = printDataLoss( d_obs, m_ini, obs_vec, simul_params ) 
    while True: 
        m_old = m_new
        f.write( "Iteration: %d \n" %q )
        m_1_new = 2 * posteriorTransProx( d_obs, m_1, m_1, obs_vec, channel_params, simul_params, pnp_params ) - m_1 
        if ( pnp_params["denoiser"] == "bm3d"): 
            m_2_new = bm3dPrior(m_2, simul_params, pnp_params)
        elif (pnp_params["denoiser"] == "TV"): 
            m_2_new = tvPrior(m_2, simul_params)
        elif (pnp_params["denoiser"] == "dncnn"):
            m_2_new = dncnnPrior(m_2, channel_params, simul_params)
        m_2_new = 2 * m_2_new - m_2
        m_3_new = 2 * vaeGeologyPrior(m_3, channel_params, simul_params, pnp_params) - m_3
        # 2G - I 
        m_1 = (1 - pnp_params["rho"]) * m_1 + pnp_params["rho"]/3 * ( -1 * m_1_new + 2 * m_2_new + 2 * m_3_new )  
        m_2 = (1 - pnp_params["rho"]) * m_2 + pnp_params["rho"]/3 * ( 2 * m_1_new + -1 * m_2_new + 2 * m_3_new ) 
        m_3 = (1 - pnp_params["rho"]) * m_3 + pnp_params["rho"]/3 * ( 2 * m_1_new + 2 * m_2_new - 1 * m_3_new ) 
        m_new = (m_1 + m_2 + m_3)/3
        res = pnpResidual( m_old, m_new )
        obj, d_new = printDataLoss( d_obs, m_new, obs_vec, simul_params ) 
        f.write("Data misfit: %6.2f, residual: %2.6f \n"%(obj,res))
        if ( res < pnp_params["tol"] or q > pnp_params["max_iter"] ): 
           break
        q += 1  
    f.write("Ensemble number %d, time : %5.2f s \n" %(ind, time()-start_time))
    f.close()
    np.savetxt( output_dir + "m_new_" + str(ind) + ".txt", m_new)

if __name__ == '__main__': 
    ind = sys.argv[1]
    ind = int(ind)
    PnP_p_d(ind)

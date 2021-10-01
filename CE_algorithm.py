###########################################################
## CE_algorithm.py: functions for implement CE algorithm ##
###########################################################

import numpy as np 
import dolfin as df 
import sys 
import os 
from time import time 
from util.misc import *
from util.plot import * 
from util.sampling import * 
from util.forward import *
from model.dataFidelity import *
from model.denoisePrior import * 
from model.initial import *
from model.dncnnPrior import * 
from model.vaePrior import *

## get true channel true field and its forward simulation result
def getChannelTrue( channel_params, simul_params, input_file, input_dir):
    m_true = getSingleField( input_file, input_dir )
    d_true = forwardWell( m_true, channel_params, simul_params)
    d_true += np.random.normal ( 0, 0.01, d_true.shape ) 
    return m_true, d_true

## compute the data misfit with observed data and true result 
def getDataLoss( d_obs, m, obs_vec, channel_params, simul_params): 
    d = forwardWell( m, channel_params, simul_params )
    nwstep = simul_params["num_wsteps"]
    obj = 0
    for n in range(nwstep  + 1):
        d_ = d[:, n:n+1]
        d_obs_ = d_obs[:, n:n+1]
        obj += 0.5 * np.dot((obs_vec*(d_ - d_obs_)).T, obs_vec*(d_ - d_obs_))
    obj /= (nwstep + 1)
    return obj, d 

## get the directory in case folder: data, log, output 
def get_case_dirs( input_dir ): 
    data_dir = input_dir + "data"
    if not os.path.isdir(data_dir): 
        sys.exit("data folder should exist in Case folder") 
    data_dir = data_dir + "/"
    log_dir = input_dir + "log"
    if not os.path.isdir(log_dir): 
        os.mkdir(log_dir) 
    output_dir = input_dir + "output" 
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_dir = output_dir + "/" 

    return data_dir, log_dir, output_dir 
    
## compute the residual between updated and previous solution to check convergence of algorithm
def getResidual(m_old, m_new):
    N = m_old.shape[0] 
    res = np.linalg.norm(m_old - m_new) / N
    return res

## one step CE algorithm with data fidelity and denoiser agents
def one_step_dat_den( m_1, m_2, d_obs, obs_vec, channel_params, simul_params, pnp_params ): 
    m_1_new = 2 * dataFidelitysProx( d_obs, m_1, m_1, obs_vec, channel_params, simul_params, pnp_params ) - m_1
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
    return m_1, m_2 

## one step CE algorithm with data fidelity, denoiser, and VAE agents 
def one_step_dat_den_vae( m_1, m_2, m_3, d_obs, obs_vec, channel_params, simul_params, pnp_params ): 
    m_1_new = 2 * dataFidelityProx( d_obs, m_1, m_1, obs_vec, channel_params, simul_params, pnp_params ) - m_1
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
   return m_1, m_2, m_3
 


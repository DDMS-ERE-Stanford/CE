# deterministic case main 

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

def getChannelTrue( channel_params, simul_params, input_file, input_dir):
    Dir = os.getcwd() + '/data/'
    m_true,_ = getInput(Dir, input_file, 1) 
    plotField(m_true, simul_params, "perm_true", input_dir)
    d_true = forwardWell( m_true, channel_params, simul_params )
    d_true += np.random.normal ( 0, 0.2, d_true.shape )  
    plotFieldTstep( d_true, simul_params, "head_output", input_dir)  
    return m_true, d_true

def posteriorObj(m, d_obs, obs_vec, channel_params, simul_params, obs_sigma): 
    d = forwardWell(m, channel_params, simul_params)
    nwstep = simul_params["num_wsteps"]
    obj = 0 
    for n in range(nwstep + 1): 
        d_ = d[:,n:n+1]
        d_obs_ = d_obs[:,n:n+1]
        obj += 0.5 * np.dot((obs_vec*(d_-d_obs_)).T, obs_vec*(d_-d_obs_)) * 1/(obs_sigma**2)
    obj /= (nwstep + 1)
    
    return d, obj
    
def pnpResidual(m_old, m_new):
    N = m_old.shape[0] 
    res = np.linalg.norm(m_old - m_new) / N
    return res

def saveOutput( m, d, simul_params, pnp_params, output_dir ):
    np.savetxt( output_dir + pnp_params["prior"] + ".txt", m )  
    plotField( m, simul_params, "perm_output", output_dir) 
    plotFieldTstep( d, simul_params, "head_output", output_dir)  

def plotHeads():  
    output_dir = os.getcwd() + '/output/'
    input_dir = os.getcwd() + '/data/' 
    filename_dncnn = output_dir + "dncnn.txt" 
    filename_TV = output_dir + "TV.txt" 
    filename_bm3d = output_dir + "bm3d.txt" 
    input_file = "channel_field.txt" 
    channel_params = { "lperm": -4, "hperm": -2, "initial": "SVR"}
    obs_params = { "nxblock_stat": 5, "nyblock_stat": 5, "nxblock_dyn": 5, "nyblock_dyn": 5 } 

    simul_params = { "ngx" : 45, "ngy" : 45, "dx": 10, "dy": 10,\
                     "lbc" : 21, "rbc" : 21, "wCond" : 1, "dt_p": 0, "dt_w": 5,\
                     "num_psteps": 0, "num_wsteps": 20, "n_wells": 1, "well_locs": [23, 23]  } 
    well_pos = []
    well_locs = simul_params["well_locs"]
    well_pos.append([well_locs[0] * simul_params["dx"], well_locs[1]* simul_params["dy"]])
   
    simul_params["dt_w"] = simul_params["dt_w"] * 3600
    simul_params["well_pos"] = well_pos 

    d_ini = 21.0 * np.ones((simul_params["ngx"] * simul_params["ngy"], 1))
    
    simul_params["ini_cond"] = d_ini
    
    m_true, d_obs = getChannelTrue( channel_params, simul_params, input_file, input_dir ) 

    m_initial = svmInitial( m_true, channel_params, simul_params, obs_params )

    d_prior = forwardWell( m_initial, channel_params, simul_params ) 

    m_dncnn = np.loadtxt( filename_dncnn )
    m_TV = np.loadtxt( filename_TV )
    m_bm3d = np.loadtxt( filename_bm3d )

    d_dncnn = forwardWell( m_dncnn, channel_params, simul_params ) 
    d_TV = forwardWellTrans( m_TV, simul_params ) 
    d_bm3d = forwardWellTrans( m_bm3d, simul_params ) 

    plotProfile( 11, 11,  d_prior, d_TV, d_bm3d,  d_dncnn, d_obs, channel_params, simul_params, "Profile1", output_dir)   
    plotProfile( 34, 11,  d_prior, d_TV, d_bm3d,  d_dncnn, d_obs, channel_params, simul_params, "Profile2", output_dir)  
    plotProfile( 11, 34,  d_prior, d_TV, d_bm3d,  d_dncnn, d_obs, channel_params, simul_params, "Profile3", output_dir)  
    plotProfile( 34, 34,  d_prior, d_TV, d_bm3d,  d_dncnn, d_obs, channel_params, simul_params, "Profile4", output_dir)  
    
def channelTrans():
    channel_params = { "lperm": -4, "hperm": -2, "initial": "SVR"}
    obs_params = { "nxblock_stat": 5, "nyblock_stat": 5, "nxblock_dyn": 5, "nyblock_dyn": 5 } 
    simul_params = { "ngx" : 45, "ngy" : 45, "dx": 10, "dy": 10,\
                     "lbc" : 21, "rbc" : 21, "wCond" : 1, "dt_p": 0, "dt_w": 5,\
                     "num_psteps": 0, "num_wsteps": 6, "n_wells": 1, "well_locs": [23, 23]  } 

    pnp_params = { "obs_sigma": 0.1, "prior_sigma": 0.4, "obs_reg": 30, "rho": 0.5, "max_iter": 0, "tol": 1e-4, "prior":"dncnn" }
    input_dir = os.getcwd() + '/data/' 
    output_dir = os.getcwd() + '/output/'
    log_dir = os.getcwd() + '/log/'
    input_file = "channel_field.txt"
 
    well_pos = []
    well_locs = simul_params["well_locs"] 
    for i in range(simul_params["n_wells"]): 
        well_pos.append([well_locs[0] * simul_params["dx"], well_locs[1]* simul_params["dy"]])
  
    simul_params["dt_w"] = simul_params["dt_w"] * 3600
    simul_params["well_pos"] = well_pos 

    d_ini = 21.0 * np.ones((simul_params["ngx"] * simul_params["ngy"], 1))
    
    simul_params["ini_cond"] = d_ini
 
    start_time = time()

    simul_params["output_steps"] = [0, 3, 5]

    # get True data 
    m_true, d_obs = getChannelTrue( channel_params, simul_params, input_file, input_dir ) 
    log_file = "log.txt"
    f = open( log_dir + log_file, "w")
 
    # construct SVM Gaussian initial 
    if (channel_params["initial"] == "Gauss"): 
        m_initial = gaussInitial( m_true, channel_params, simul_params, obs_params )
    elif (channel_params["initial"] == "SVR"):
        m_initial = svmInitial( m_true, channel_params, simul_params, obs_params )

    # dynamic data sampling vector
    _, obs_vec = obsMatrix( obs_params["nxblock_dyn"], obs_params["nyblock_dyn"], simul_params["ngx"], simul_params["ngy"] )
 
    # initial loss 
    _, obj = posteriorObj( m_initial, d_obs, obs_vec, channel_params, simul_params, pnp_params["obs_sigma"] )

    f.write('Initial loss: %5.2f' %obj)
    m_new = m_initial
    m_1 = m_initial 
    m_2 = m_initial
    q = 1
    f.write('----- Start PnP-DR Algorithm ----- \n')

    while True: 
        m_old = m_new

        # posterior proximl mapping
        f.write("Iteartion: %d \n" %q ) 
        temp_time = time()
        m_1_new = posteriorTransProx(d_obs, m_1, m_1, obs_vec, channel_params, simul_params,  pnp_params)  
        m_1_new = 2 * m_1_new - m_1
        f.write("posterior time: %5.2f s" %(time() - temp_time))
        temp_time = time()

        # denoise prior
        if (pnp_params["prior"] == "bm3d"):
            m_2_new = bm3dPrior(m_2, simul_params, pnp_params)
        elif (pnp_params["prior"] == "TV"):
            m_2_new = tvPrior(m_2, simul_params)
        elif (pnp_params["prior"] == "dncnn"): 
            m_2_new = dncnnPrior(m_2, channel_params, simul_params)
        m_2_new = 2 * m_2_new - m_2 
        f.write("denoiser time: %5.2f s" %(time() - temp_time))

        # 2G - I 
        m_1 = (1 - pnp_params["rho"]) * m_1 + pnp_params["rho"] * m_2_new
        m_2 = (1 - pnp_params["rho"]) * m_2 + pnp_params["rho"] * m_1_new 
        m_new = (m_1 + m_2)/2
        d, obj = posteriorObj( m_new, d_obs, obs_vec, channel_params, simul_params, pnp_params["obs_sigma"]) 
        res = pnpResidual( m_old, m_new )
        f.write("Data misfit: %6.2f, residual: %2.6f"%(obj,res))
        if ( res < pnp_params["tol"] or q > pnp_params["max_iter"] ): 
            break
        q += 1
   
    saveOutput( m_new, d, simul_params, pnp_params, output_dir )
    f.write("---- Finish prgram(Total time : %5.2f s) ----" %(time() -start_time))
    
    obs_x, obs_y = obsLoc( obs_params["nxblock_dyn"], obs_params["nyblock_dyn"], simul_params["ngx"], simul_params["ngy"] )
    plotFieldWell(m_true, simul_params, obs_x, obs_y, "True_well", output_dir) 

if __name__ == '__main__': 
    #plotHeads()
    channelTrans()

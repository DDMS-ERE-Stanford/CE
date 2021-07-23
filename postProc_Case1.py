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
from model.vaePrior2 import *
import shutil
import multiprocessing 
from functools import partial 

home_dir = "/data/cees/hjyang3/PnP/"
folder = "Case4/"
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
#    m_true *= (channel_params["hperm"] - channel_params["lperm"]) 
#    m_true += channel_params["lperm"] 
    d_true = forwardWell( m_true, channel_params, simul_params)
    return m_true, d_true

def dataMisfit( d, d_obs, obs_vec, simul_params): 
    nstep = simul_params["num_wsteps"]
    obj = 0
    for n in range(nstep+1):
        d_ = d[:, n:n+1]
        d_obs_ = d_obs[:, n+1:n+2]
        obj += 0.5 * np.dot((obs_vec*(d_-d_obs_)).T, obs_vec*(d_-d_obs_))
    obj /= nstep
    return obj   

def printDataLoss( d_obs, m, obs_vec, simul_params): 
    d = forwardWell( m, channel_params, simul_params)
    obj = dataMisfit( d, d_obs, obs_vec, simul_params )
    return obj, d 
  
def pnpResidual(m_old, m_new):
    N = m_old.shape[0] 
    res = np.linalg.norm(m_old - m_new) / N
    return res

def PSNR(m_true, m_update): 
    N = m_true.shape[0] 
    mse = (np.square(m_true-m_update)).mean(axis = 1)
    MAX_I = np.max(m_update)
    psnr = 20 * np.log10(MAX_I) - 10 * np.log10(mse) 
    return psnr, mse 

def threshold(m_true, m_update, channel_params): 
    threshold = 3.0
    m_thres = (m_update > threshold).astype(float) 
    m_thres *= channel_params["hperm"] 
    misclass = np.abs(m_true - m_thres)
    misclass_ = (misclass > 1e-1).astype(int) 
    N_mis = np.sum(misclass_) 
    return m_thres, N_mis 
       
def MPS():
    n_samples = channel_params["num_ens"] 
    m_true, d_obs = getChannelTrue( channel_params, simul_params, Ref_file, data_dir)
    #samples = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 126, 127, 128, 129, 130]
    m_new_final = np.zeros((simul_params["ngx"] * simul_params["ngy"], n_samples))
    m_c_final = np.zeros((simul_params["ngx"] * simul_params["ngy"], n_samples))
    d_new_final = np.zeros((simul_params["ngx"] * simul_params["ngy"], simul_params["num_wsteps"]+1, n_samples))
    m_ini_final = np.zeros((simul_params["ngx"] * simul_params["ngy"], n_samples))
    d_ini_final = np.zeros((simul_params["ngx"] * simul_params["ngy"], simul_params["num_wsteps"]+1, n_samples))
    result_file = "result.txt"
    f = open( output_dir + result_file, "w" )

    for ind in range(n_samples):
        filename = output_dir + "m_new_" + str( ind + 51 ) + ".txt"
    #    filename2 = output_dir + "m_c_" + str( ind + 100 ) + ".txt"
        m_new_final[:, ind] = np.loadtxt( filename )
       # m_c_final[:, ind] = np.loadtxt( filename2 )
        m_ini_final[:,ind:ind+1] = getMPSInitial( ind + 51, channel_params, simul_params, MPS_file, data_dir ) 
        d_new_final[:,:,ind] = forwardWell( m_new_final[:, ind], channel_params, simul_params )
        d_ini_final[:,:,ind] = forwardWell( m_ini_final[:, ind], channel_params, simul_params )

    plotUncertainty(20, 20, d_new_final, d_obs, channel_params, simul_params, 'uncertainty1', output_dir)
    plotUncertainty(20, 60, d_new_final, d_obs, channel_params, simul_params, 'uncertainty2', output_dir)
    plotUncertainty(60, 20, d_new_final, d_obs, channel_params, simul_params, 'uncertainty3', output_dir)
    plotUncertainty(60, 60, d_new_final, d_obs, channel_params, simul_params, 'uncertainty4', output_dir)

    plotUncertainty(20, 20, d_ini_final, d_obs, channel_params, simul_params, 'uncertainty1_ini', output_dir)
    plotUncertainty(20, 60, d_ini_final, d_obs, channel_params, simul_params, 'uncertainty2_ini', output_dir)
    plotUncertainty(60, 20, d_ini_final, d_obs, channel_params, simul_params, 'uncertainty3_ini', output_dir)
    plotUncertainty(60, 60, d_ini_final, d_obs, channel_params, simul_params, 'uncertainty4_ini', output_dir)


    m_ini_avg = np.mean( m_ini_final, axis = 1, keepdims = True )
    m_new_avg = np.mean( m_new_final, axis = 1, keepdims = True )
  #  m_c_avg = np.mean( m_c_final, axis = 1, keepdims = True )
    m_thres, N_mis = threshold(m_true, m_new_avg, channel_params) 
 
#    f.write("Misclassified cells: %d \n" %N_mis)
#    f.close()
 #   d_new_avg = forwardWellTrans( m_new_avg, simul_params )
 #   d_ini_avg = forwardWellTrans( m_ini_avg, simul_params )

 #   plotField( m_thres, simul_params, "update_threshold", output_dir )
    perm_ini_avg = m_ini_avg *  (channel_params["hperm"] - channel_params["lperm"]) + channel_params["lperm"]
    perm_true = m_true *  (channel_params["hperm"] - channel_params["lperm"]) + channel_params["lperm"]
    perm_new = m_new_avg *  (channel_params["hperm"] - channel_params["lperm"]) + channel_params["lperm"]

   # perm_c_avg = m_c_avg *  (channel_params["hperm"] - channel_params["lperm"]) + channel_params["lperm"]

    plotField( perm_ini_avg, simul_params, "initial_field_avg", output_dir )
   # plotField( perm_c_avg, simul_params, "compare_field_avg", output_dir )
    plotField( perm_new, simul_params, "updated_field_avg", output_dir )
    plotField( perm_true, simul_params, "true_field", output_dir ) 
    plotFieldTstep( d_obs, simul_params, "true_head", output_dir )     
#    plotFieldTstep( d_ini_avg, simul_params, "initial_avg_head", output_dir )    
#    plotFieldTstep( d_new_avg, simul_params, "updated_avg_head", output_dir )   
 
if __name__ == '__main__':
    MPS() 

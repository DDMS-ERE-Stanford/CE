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
   # m_true *= (channel_params["hperm"] - channel_params["lperm"]) 
   # m_true += channel_params["lperm"] 
    d_true = forwardWell( m_true, channel_params, simul_params)
    return m_true, d_true
          
def main():
    log_file = "log_true.txt" 
    f = open( log_dir + log_file, "w" )
    start_time = time() 
    f.write("--- Simulation of True field ---- \n")
    obs_x, obs_y = obsLoc ( obs_params["nxblock_dyn"], obs_params["nyblock_dyn"], simul_params["ngx"], simul_params["ngy"]);

    m_true, d_obs = getChannelTrue( channel_params, simul_params, Ref_file, data_dir)
    m_ini = getMPSInitial( 2, channel_params, simul_params, MPS_file, data_dir )

    plotFieldWell( m_true, simul_params, obs_x, obs_y, "true_field_well", output_dir ) 
    plotField( m_ini, simul_params, "initial_field", output_dir ) 
    plotField( m_true, simul_params, "true_field", output_dir )
    plotFieldTstep( d_obs, simul_params, "true_head", output_dir )
    f.write("---- Finish prgram(Total time : %5.2f s) ---- \n" %(time() -start_time))
    f.close()

if __name__ == '__main__': 
    main()

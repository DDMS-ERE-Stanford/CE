##########################################################
## Util functions for reading input ######################
##########################################################

import os 
import math
import numpy as np 

def getSingleField(filename, Dir):
   out_ = np.loadtxt(Dir+filename, dtype = 'float')
   out = out_.reshape(( -1, 1))
   outTrue = out[:, 0:1]
   return outTrue

def getMultiFields(N_grid, filename, Dir): 
    out_ = np.loadtxt(Dir+filename, dtype = 'float')
    out = out_.reshape((-1, N_grid))
    return out

def output(Dir, filename, out):
    np.savetxt(Dir + filename,  out.flatten(), delimiter='\n')

def readSimulParams(infile): 
    infile.readline() 
    line = infile.readline()
    ce_type = int(line) 

    infile.readline()
    line = infile.readline()
    ngx, ngy = line.split() 

    infile.readline() 
    line = infile.readline() 
    dx, dy = line.split() 

    infile.readline() 
    line = infile.readline() 
    d_ini, lbc, rbc, wCond = line.split()

    infile.readline() 
    line = infile.readline() 
    n_wells = int(line) 

    infile.readline() 
    well_locs = []
    well_pos = []
    for i in range(n_wells):
        line = infile.readline()
        x_loc, y_loc = line.split()
        well_locs.append([int(x_loc),int(y_loc)])
    
    infile.readline()
    line = infile.readline() 
    dt_p, num_psteps = line.split()

    infile.readline() 
    line = infile.readline() 
    dt_w, num_wsteps = line.split()

    infile.readline() 
    line = infile.readline()
    output_steps = line.split()
    output_steps = [int(i) for i in output_steps]

     
    simul_params = { "ce_type": int(ce_type), "ngx" : int(ngx), "ngy" : int(ngy), "dx": float(dx), "dy": float(dy),\
                     "ini_cond": float(d_ini), "lbc": float(lbc), "rbc": float(rbc), "wCond": float(wCond),\
                     "n_wells": n_wells, "well_locs": well_locs, \
                     "dt_p": float(dt_p),"dt_w": float(dt_w), "num_psteps": int(num_psteps), "num_wsteps": int(num_wsteps),\
                     "output_steps": output_steps }                  

    return infile, simul_params

def unitConversion( simul_params, channel_params ):
    # convert time unit and set well pos  
    ngx = simul_params["ngx"] 
    ngy = simul_params["ngy"]
    well_locs = simul_params["well_locs"]
    well_pos = []
    for i in range(simul_params["n_wells"]):
        well_pos.append([well_locs[i][0]*simul_params["dx"], well_locs[i][1]*simul_params["dy"]])
    simul_params["well_pos"] = well_pos
    simul_params["dt_p"] = simul_params["dt_p"] * 3600 * 24 #days to secs
    simul_params["dt_w"] = simul_params["dt_w"] * 3600 * 24 
    return simul_params, channel_params
    
def readChannelParams(infile):
    infile.readline()
    line = infile.readline() 
    true_file = line.strip()
    infile.readline() 
    line = infile.readline() 
    lperm, hperm, initial = line.split()
    channel_params = { "true_file": true_file, "lperm" : float(lperm), "hperm" : float(hperm), "initial": initial }
    if (initial == "MPS"):
       infile.readline() 
       line = infile.readline() 
       channel_params["mps_file"] = line.strip() 
       infile.readline() 
       line = infile.readline()
       channel_params["num_ens"] = int(line)
       infile.readline() 
       line = infile.readline() 
       channel_params["total_samples"] = int(line) 
    return infile, channel_params

def readObsParams(infile): 
    infile.readline() 
    line = infile.readline() 
    nxblock_dyn, nyblock_dyn, nxblock_stat, nyblock_stat = line.split() 
    obs_params = { "nxblock_dyn": int(nxblock_dyn), "nyblock_dyn": int(nyblock_dyn),\
                   "nxblock_stat": int(nxblock_stat), "nyblock_stat": int(nyblock_stat) }
    return infile, obs_params

def readPnPParams(infile):

    infile.readline() 
    line = infile.readline() 
    rho, max_iter, tol = line.split() 
 
    infile.readline() 
    line = infile.readline() 
    num_models = line 
    infile.readline()
    line = infile.readline() 
    obs_sigma, obs_reg = line.split()
    pnp_params = {"rho": float(rho), "max_iter" : int(max_iter), "tol" : float(tol), \
                  "num_models" : int(num_models), "obs_sigma": float(obs_sigma), "obs_reg": float(obs_reg) }

    if (pnp_params["num_models"] > 1) : 
        infile.readline() 
        line = infile.readline()
        dn_sigma, denoiser = line.split()
        pnp_params["dn_sigma"] = float(dn_sigma) 
        pnp_params["denoiser"] = denoiser

    elif (pnp_params["num_models"] > 2) : 
        infile.readline() 
        line = infile.readline() 
        vae_reg = line 
        pnp_params["vae_reg"] = float(vae_reg)

    else: 
        sys.exit("Number of PnP models should be greater than 1")

    return infile, pnp_params

def readInput(filename, Dir):
    infile = open(Dir + filename, "r") 
    infile, simul_params = readSimulParams(infile)
    infile, channel_params = readChannelParams(infile)
    simul_params, channel_params = unitConversion(simul_params, channel_params)
    infile, obs_params = readObsParams(infile)
    infile, pnp_params = readPnPParams(infile)
    infile.close()

    return simul_params, channel_params, obs_params, pnp_params


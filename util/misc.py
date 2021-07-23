import os 
import math
import numpy as np 
import h5py
from mpi4py import MPI as mpi 

comm = mpi.COMM_WORLD
rank = comm.Get_rank() 

def getInput(Dir, filename, numOfSamples):
   out_ = np.loadtxt(Dir+filename, dtype = 'float')
   out = out_.reshape(( -1, numOfSamples))
   outTrue = out[:, 0:1]
   if numOfSamples > 1: 
      outEn = out[:, 1:]
   else : 
      outEn = None;
   return outTrue, outEn

def getEns(N_grid, filename, Dir): 
    out_ = np.loadtxt(Dir+filename, dtype = 'float')
    out = out_.reshape((-1, N_grid))
    return out

def output(Dir, filename, out):
    np.savetxt(Dir + filename,  out.flatten(), delimiter='\n')

def readSimulParams(infile): 
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

     
    infile.readline()
    num_cores = infile.readline() 
    simul_params = { "ngx" : int(ngx), "ngy" : int(ngy), "dx": float(dx), "dy": float(dy),\
                     "ini_cond": float(d_ini), "lbc": float(lbc), "rbc": float(rbc), "wCond": float(wCond),\
                     "n_wells": n_wells, "well_locs": well_locs, \
                     "dt_p": float(dt_p),"dt_w": float(dt_w), "num_psteps": int(num_psteps), "num_wsteps": int(num_wsteps),\
                     "output_steps": output_steps }                  

    simul_params["num_cores"] = int(num_cores)
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
    lperm, hperm, initial = line.split()
    channel_params = { "lperm" : float(lperm), "hperm" : float(hperm), "initial": initial }
    if (initial == "MPS"): 
       infile.readline() 
       line = infile.readline()
       channel_params["num_ens"] = int(line) 
    return infile, channel_params

def readObsParams(infile): 
    infile.readline() 
    line = infile.readline() 
    nxblock_dyn, nyblock_dyn, nxblock_stat, nyblock_stat = line.split() 
    obs_params = { "nxblock_dyn": int(nxblock_dyn), "nyblock_dyn": int(nyblock_dyn), "nxblock_stat": int(nxblock_stat), "nyblock_stat": int(nyblock_stat)}
    return infile, obs_params

def readPnPParams(infile):
    infile.readline() 
    line = infile.readline() 
    num_models = line 
    infile.readline()
    line = infile.readline() 
    obs_sigma, obs_reg = line.split()
    pnp_params = { "num_models" : int(num_models), "obs_sigma": float(obs_sigma), "obs_reg": float(obs_reg) }
    if (pnp_params["num_models"] > 1) : 
        infile.readline() 
        line = infile.readline()
        dn_sigma, denoiser = line.split()
        pnp_params["dn_sigma"] = float(dn_sigma) 
        pnp_params["denoiser"] = denoiser

    if (pnp_params["num_models"] > 2) : 
        infile.readline() 
        line = infile.readline() 
        vae_reg = line 
        pnp_params["vae_reg"] = float(vae_reg)

    infile.readline() 
    line = infile.readline() 
    rho, max_iter, tol = line.split() 
    pnp_params["rho"] = float(rho) 
    pnp_params["max_iter"] = int(max_iter) 
    pnp_params["tol"] = float(tol) 
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

def readHDF(filename, Dir): 
    hf = h5py.File(Dir + filename + '.h5','r')
    n1 = hf['Y']
    out = np.array(n1["vector_0"])
    out = out[:, np.newaxis]
    hf.close() 
    return out

def updateHDF(m, filename, Dir): 
    hf = h5py.File(Dir + filename + '.h5','r+')
    n1 = hf['Y']
    n1["vector_0"][:] = m.flatten()
    hf.close() 
 
def readOutParallel(num_cores, filename, Dir): 
    out = np.empty((0,3)) 
    for i in range(num_cores): 
        name = Dir + filename + str(i) +'.txt'
        temp = np.loadtxt(name)
        out = np.append( out, temp, axis = 0 )
    ind = np.lexsort((out[:,0], out[:,1]))
    comm.Barrier() 
    if rank == 0: 
        delAuxFiles( num_cores, filename, Dir)
    return out[ind, 2:3]

def readOutTstepParallel(num_cores, num_tsteps, filename, Dir): 
    out = np.empty((0, 2 + num_tsteps)) 
    for i in range(num_cores): 
        name = Dir + filename + str(i) +'.txt'
        temp = np.loadtxt(name)
        out = np.append( out, temp, axis = 0 )
    ind = np.lexsort((out[:,0], out[:,1]))
    comm.Barrier() 
    if rank == 0: 
        delAuxFiles( num_cores, filename, Dir )
    return out[ind, 2:2+num_tsteps]

def delAuxFiles( num_cores, filename, Dir ): 
    for i in range(num_cores): 
        name = Dir + filename + str(i) + '.txt'
        os.remove(name) 

def main():
    Dir = "/data/cees/hjyang3/PnP/Case1/"
    filename = "Input.txt"
    simul_params, channel_params, pnp_params = readInput(filename, Dir) 
    print(simul_params)
    print(channel_params)
    print(pnp_params)

if __name__ == '__main__':
    main()

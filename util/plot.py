#################################################
## plot.py: util function for plotting the results
##################################################


import matplotlib.pyplot as plt
import numpy as np
import sys 
import os 

## plotField: plot one field 
def plotField(u, simul_params, Filename, Dir): 
    ngx = simul_params["ngx"]
    ngy = simul_params["ngy"]
    u = u.reshape(ngy,ngx) 
    plt.figure()
    plt.imshow(u)
    plt.colorbar()
    plt.savefig(Dir + Filename+'.png')

## plotFieldWell: plot field with observation and production/injection wells  
def plotFieldWell(u, simul_params, obs_x, obs_y, Filename, Dir): 
    ngx = simul_params["ngx"]
    ngy = simul_params["ngy"]
    u = u.reshape(ngy,ngx) 
    plt.figure()
    plt.imshow(u)
    plt.colorbar()
    plt.scatter(obs_x, obs_y, marker = '+', color ='white', s = 80)
    well_pos = simul_params["well_pos"]
    for i in range(simul_params["n_wells"]):
        plt.scatter( well_pos[i][0]/simul_params["dx"], well_pos[i][1]/simul_params["dy"], color = 'red', s = 100 )
    plt.savefig(Dir + Filename+'.png')


## plotProfile: plot for comparing the evolution of QOI
def plotProfile(i, j, u_ini, u_1, u_2, u_3, u_true, channel_params,simul_params, Filename, Dir): 
    ngx = simul_params["ngx"]
    ngy = simul_params["ngy"]
    nsteps = simul_params["num_wsteps"]
    x1 = np.linspace(0, 50, nsteps+1, endpoint = True)
    ind = (i-1) + (j-1) * ngx 
    qoi_1 = u_1[ind, :] 
    qoi_2 = u_2[ind, :] 
    qoi_3 = u_3[ind, :] 
    qoi_ini = u_ini[ind, :] 
    qoi_true = u_true[ind, :]
    plt.figure()
    plt.ylim(0, 1)
    plt.xlim(0, 50)
    plt.plot(x1, qoi_ini, color = 'gray', linewidth = 3)
    plt.plot(x1, qoi_1, color = 'r', linestyle = ':', linewidth = 3)
    plt.plot(x1, qoi_2, color = 'r', linestyle = '--', linewidth = 3)
    plt.plot(x1, qoi_3, color = 'r', linestyle = '-', linewidth = 3)
    plt.plot(x1, qoi_true, color = 'blue', linewidth = 3)
    plt.savefig(Dir + Filename + '.png')  

   
## plotUncerainty: plot the uncertainty ranges of ensemble results and true result
def plotUncertainty(i, j, u, u_true, channel_params,simul_params, Filename, Dir): 
    ngx = simul_params["ngx"]
    ngy = simul_params["ngy"]
    output_tsteps = simul_params["output_steps"]
    nsteps = simul_params["num_wsteps"]
    n_samples = channel_params["num_ens"]
    x1 = np.linspace(0, 200, nsteps+1, endpoint = True)
    ind = (i-1) + (j-1) * ngx 
    qoi = u[ind, :, :] 
    qoi_true = u_true[ind, :]
    plt.figure()
    for i in range (n_samples): 
        plt.plot(x1, qoi[:, i], color = 'gray', linewidth=1)
    plt.ylim(8.0, 10.0)
    qoi_avg= np.mean(qoi, axis = 1 )
    qoi_std = np.std(qoi, axis = 1 ) 
    ci = 1.44 * qoi_std
    plt.plot(x1, qoi_true, color = 'blue', linewidth = 3)
    plt.plot(x1, qoi_avg, color = 'r', linewidth = 3)
    plt.fill_between(x1, (qoi_avg - ci), (qoi_avg+ci), alpha=.2)
    plt.savefig(Dir + Filename + '.png')  

## plotFieldTstep: plot the evolution of result field with time
def plotFieldTstep(u, simul_params, Filename, Dir): 
    ngx = simul_params["ngx"]
    ngy = simul_params["ngy"]
    output_tsteps = simul_params["output_steps"]
    for tstep in output_tsteps:
        u_plot = u[:, tstep]
        u_plot = u_plot.reshape(ngy,ngx) 
        plt.figure()
        plt.imshow(u_plot)
        plt.colorbar()
        plt.clim(1.0, 21.0)
        Filename_ = Filename + "_T_%s" % tstep
        plt.savefig(Dir + Filename_ +  '.png')
   
def plotPoints(d_obs, obs_x, obs_y, Filename, Dir): 
    plt.figure()
    plt.scatter(obs_x, obs_y, c=d_obs)
    plt.gca().invert_yaxis() 
    plt.colorbar()
    plt.savefig(Dir + Filename + '.png')  

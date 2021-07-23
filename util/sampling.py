import numpy as np
import math 
import sys
import os 
from time import time

def obsMatrix(nxblock, nyblock, ngx, ngy): 
    nObs = nxblock * nyblock 
    nGrid = ngx * ngy
    del_nx = int(ngx/(nxblock))
    del_ny = int(ngy/(nyblock))
    obs_mat = np.zeros((nObs, nGrid))
    obs_vec = np.zeros((ngx*ngy, 1))
    k = 0
    for i in range(nxblock):
        for j in range(nyblock):
            x_loc = int((del_nx-1)/2) + i * del_nx
            y_loc = int((del_ny-1)/2) + j * del_ny
            q = x_loc + (y_loc) * ngx
            obs_vec[q] = 1
            obs_mat[k, q] = 1
            k = k + 1

    return obs_mat, obs_vec

def obsLoc(nxblock, nyblock, ngx, ngy): 
    nObs = nxblock * nyblock  
    nGrid = ngx * ngy
    del_nx = int(ngx/(nxblock))
    del_ny = int(ngy/(nyblock))
    x_loc = np.zeros((nObs, 1))
    y_loc = np.zeros((nObs, 1))
    k = 0
    pts_obs = np.zeros((ngy*ngx,1)) 
    for i in range(nxblock):
       for j in range(nyblock): 
           x_loc[k] = int((del_nx-1)/2) + i * del_nx
           y_loc[k] = int((del_ny-1)/2) + j * del_ny
           k = k+1
           
    return x_loc, y_loc


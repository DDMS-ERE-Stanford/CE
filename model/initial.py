import numpy as np 
import sys
import os 
from util.sampling import * 
from sklearn.svm import SVR 
from util.plot import * 
from util.misc import *
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def getMPSInitial( initial_num, channel_params, simul_params, filename, input_dir ): 
    # get initials from external file 
    ngx = simul_params["ngx"]
    ngy = simul_params["ngy"]
#    lperm = channel_params["lperm"]
#    hperm = channel_params["hperm"]    
    Ens = getEns(ngx*ngy, filename, input_dir) 
    initials = Ens[initial_num-1, :]
    initials = initials[:, np.newaxis]
#    initials *= (hperm - lperm)
#    initials += lperm 
    return initials 

def svmInitial( true_field, channel_params, simul_params, obs_params ): 
    nxblock = obs_params["nxblock_stat"]
    nyblock = obs_params["nyblock_stat"]
    ngx = simul_params["ngx"]
    ngy = simul_params["ngy"] 
    lperm = channel_params["lperm"]
    hperm = channel_params["hperm"] 

    obs_mat, _ = obsMatrix(nxblock, nyblock, ngx, ngy) 
    data = np.matmul(obs_mat, true_field) 
    x_loc, y_loc = obsLoc(nxblock, nyblock, ngx, ngy) 
    X_train = np.hstack([x_loc, y_loc])
    Y_train = data.ravel() 
    x_grid, y_grid = np.mgrid[0:ngx:1, 0:ngy:1]
    x_grid = x_grid.T.reshape(ngx*ngy,1)
    y_grid = y_grid.T.reshape(ngx*ngy,1)
    X_test = np.hstack([x_grid, y_grid])
    SVRmodel = SVR(C= 120) 
    SVRmodel.fit(X_train, Y_train) 
    Y_pred = SVRmodel.predict(X_test)
    Y_pred = Y_pred[:, np.newaxis]

    Y_pred[Y_pred > 1 ] = 1
    Y_pred[Y_pred < 0 ] = 0
    return Y_pred

def gaussInitial( true_field, channel_params, simul_params, obs_params ): 
    nxblock = obs_params["nxblock_stat"]
    nyblock = obs_params["nyblock_stat"]
    ngx = simul_params["ngx"]
    ngy = simul_params["ngy"] 
    lperm = channel_params["lperm"]
    hperm = channel_params["hperm"] 

    obs_mat, _ = obsMatrix(nxblock, nyblock, ngx, ngy) 
    data = np.matmul(obs_mat, true_field) 
    x_loc, y_loc = obsLoc(nxblock, nyblock, ngx, ngy) 
    X_train = np.hstack([x_loc, y_loc])
    Y_train = data.ravel() 
    x_grid, y_grid = np.mgrid[0:ngx:1, 0:ngy:1]
    x_grid = x_grid.T.reshape(ngx*ngy,1)
    y_grid = y_grid.T.reshape(ngx*ngy,1)
    X_test = np.hstack([x_grid, y_grid])
    kernel = C(1.0,(1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 20)
    gp.fit(X_train, Y_train)
    Y_pred = gp.predict(X_test)
    Y_pred = Y_pred[:, np.newaxis]

    Y_pred[Y_pred > hperm ] = hperm
    Y_pred[Y_pred < lperm ] = lperm

    return Y_pred


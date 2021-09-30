#-----------------------------------------
# agent to minimize data misfit
#------------------------------------------

import numpy as np 
import sys 
import os 
from dolfin import *
from dolfin_adjoint import * 
pdir = os.path.dirname(os.getcwd()) 
sys.path.insert(0, pdir)
from time import time
import math
from util.misc import * 
from util.forward import *

## proximal operation of data fidelity function
def posteriorTransProx(d_obs_,  m_prior_, m_, obs_vec_, channel_params, simul_params, pnp_params): 
   
   ngy = simul_params["ngy"]
   ngx = simul_params["ngx"] 
   dx_ = simul_params["dx"] 
   dy_ = simul_params["dy"] 

   num_psteps = simul_params["num_psteps"]
   num_wsteps = simul_params["num_wsteps"]

   m_prior_ = m_prior_ * (channel_params["hperm"] - channel_params["lperm"]) + channel_params["lperm"]
   m_ = m_ * (channel_params["hperm"] - channel_params["lperm"]) + channel_params["lperm"]

   d_ini = simul_params["ini_cond"] * np.ones((ngy * ngx, 1))
   obs_sigma = pnp_params["obs_sigma"]
   reg = pnp_params["obs_reg"] 
   m_lb = channel_params["lperm"]
   m_ub = channel_params["hperm"]
  
   mesh = RectangleMesh(Point(0.0, 0.0), Point(ngx*dy_, ngy*dy_), ngy - 1, ngx - 1)

   V = FunctionSpace(mesh, "Lagrange", 1) 
   m = projectFunction(V, m_) 
   obs_vec = projectFunction(V, obs_vec_ ) 
   m_prior = projectFunction(V, m_prior_ ) 
   obs_sigma = Constant(obs_sigma) 
   reg = Constant(reg) 
   J = 0
   d_old = projectFunction(V, d_ini)
 
   # pre-well simulation
   for n in range(num_psteps): 
       d = forwardPrevFunc( d_old, m, V, simul_params ) 
       d_old = d  
   d_obs = projectFunction(V, d_obs_[:, 0:1] )
   J += assemble( 0.5 * 1/(obs_sigma**2) * inner(obs_vec*(d_old - d_obs), obs_vec*(d_old - d_obs)) *dx )
 
   # after-well simulation
   for n in range(num_wsteps):
       d = forwardWellFunc(d_old, m, V, simul_params ) 
       d_old = d 
       d_obs = projectFunction(V, d_obs_[:, n+1:n+2] )
       J += assemble( 0.5 * 1/(obs_sigma**2) * inner(obs_vec*(d - d_obs), obs_vec*(d - d_obs)) *dx ) 
   J /= (num_wsteps+1)

   J += assemble( 0.5 * 1/reg * inner( m - m_prior, m - m_prior)  * dx ) 
   
   control = Control(m) 
   reduced_functional = ReducedFunctional(J, control)   
   
   m_opt = minimize(reduced_functional, bounds = (m_lb, m_ub), options = {"disp":False}, tol=1e-4)

   m_out = m_opt.compute_vertex_values(mesh).reshape(ngy*ngx, 1) 
   m_out = (m_out - channel_params["lperm"])/(channel_params["hperm"] - channel_params["lperm"])

   return m_out


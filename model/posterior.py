import matplotlib.pyplot as plt
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
def eval_cb(j, m): 
    # callback function for the optimizer 
    print("objective = %15.10e" % j)

def posterior(d_obs_, m_, simul_params, obs_vec, obs_sigma): 
   ngy = simul_params["ngy"]
   ngx = simul_params["ngx"] 
    
   mesh = UnitSquareMesh(ngy - 1, ngx - 1) 

   V = FunctionSpace(mesh, "Lagrange", 1) 
   m = projectFunction(V, m_) 
   obs_vec = projectFunction(V, obs_vec_) 
   d_obs = projectFunction(V, d_obs_) 

   d = forwardWellFunc( m, V, simul_params ) 
  
   obs_sigma = Constant(obs_sigma) 
   J = assemble( 0.5 * 1/(obs_sigma**2) * inner(obs_vec*(d- d_obs), obs_vec*(d- d_obs)) *dx ) 
    
   control = Control(m) 
   reduced_functional = ReducedFunctional(J, control)   
   
   m_opt = minimize(reduced_functional, options = {"disp":False}, tol=1e-5)

   return m_opt.compute_vertex_values(mesh).reshape(ngy*ngx, 1) 
  
def posterior_prox(d_obs_, m_prior_, m_, obs_vec_, channel_params, simul_params, pnp_params): 
   
   ngy = simul_params["ngy"]
   ngx = simul_params["ngx"] 
   obs_sigma = pnp_params["obs_sigma"]
   reg = pnp_params["reg"] 
   m_lb = channel_params["lperm"]
   m_ub = channel_params["hperm"]
   
   mesh = UnitSquareMesh(ngy - 1, ngx - 1) 

   V = FunctionSpace(mesh, "Lagrange", 1) 
   m = projectFunction(V, m_) 
   obs_vec = projectFunction(V, obs_vec_) 
   d_obs = projectFunction(V, d_obs_ ) 
   m_prior = projectFunction(V, m_prior_) 

   d = forwardWellFunc( m, V, simul_params ) 
  
   obs_sigma = Constant(obs_sigma) 
   reg = Constant(reg) 
   J = assemble( 0.5 * 1/(obs_sigma**2) * inner(obs_vec*(d- d_obs), obs_vec*(d- d_obs)) *dx ) 
   J += assemble( 0.5 * 1/reg * inner( m- m_prior, m-m_prior)  * dx ) 
    
   control = Control(m) 
   reduced_functional = ReducedFunctional(J, control)   
   
   m_opt = minimize(reduced_functional, bounds = (m_lb, m_ub), options = {"disp":False}, tol=1e-4)

   return m_opt.compute_vertex_values(mesh).reshape(ngy*ngx, 1) 

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

def posteriorTransProx2(d_obs_,  m_prior_, m_, obs_vec_, channel_params, simul_params, pnp_params): 
   
   ngy = simul_params["ngy"]
   ngx = simul_params["ngx"] 
   num_steps = simul_params["num_steps"]
   d_ini = simul_params["ini_cond"] * np.ones((ngy * ngx, 1))
   obs_sigma = pnp_params["obs_sigma"]
   reg = pnp_params["obs_reg"] 
   m_lb = 0
   m_ub = 1
  
   mesh = UnitSquareMesh(ngy - 1, ngx - 1) 

   V = FunctionSpace(mesh, "Lagrange", 1) 
   m = projectFunction(V, m_) 
   obs_vec = projectFunction(V, obs_vec_ ) 
   m_prior = projectFunction(V, m_prior_ ) 
   obs_sigma = Constant(obs_sigma) 
   reg = Constant(reg) 
   J = 0
   d_old = projectFunction(V, d_ini)
   for n in range(num_steps):
       d = forwardWellTransFunc2(d_old, m, V, simul_params ) 
       d_old = d 
       d_obs = projectFunction(V, d_obs_[:, n+1:n+2] )
       J += assemble( 0.5 * 1/(obs_sigma**2) * inner(obs_vec*(d- d_obs), obs_vec*(d- d_obs)) *dx ) 
   J /= num_steps
   J += assemble( 0.5 * 1/reg * inner( m- m_prior, m-m_prior)  * dx ) 
   
   control = Control(m) 
   reduced_functional = ReducedFunctional(J, control)   
   
   m_opt = minimize(reduced_functional, bounds = (m_lb, m_ub), options = {"disp":False}, tol=1e-4)

   return m_opt.compute_vertex_values(mesh).reshape(ngy*ngx, 1)

def posteriorProxParallel(d_out_file, m_prior_file, obs_vec_file, channel_params, simul_params, pnp_params, Dir): 
   
   ngy = simul_params["ngy"]
   ngx = simul_params["ngx"] 
   num_steps = simul_params["num_steps"]
   d_ini = simul_params["ini_cond"]
   num_cores = simul_params["num_cores"] 
   obs_sigma = pnp_params["obs_sigma"]
   reg = pnp_params["obs_reg"] 
   m_lb = channel_params["lperm"]
   m_ub = channel_params["hperm"]
  
   mesh = UnitSquareMesh(ngy - 1, ngx - 1) 
   mpi_comm = mesh.mpi_comm() 
   
   V = FunctionSpace(mesh, "Lagrange", 1) 
   m = projectHDF( mpi_comm, V, m_prior_file, Dir ) 
   m_prior = projectHDF( mpi_comm, V,  m_prior_file, Dir ) 
   obs_vec = projectHDF( mpi_comm, V,  obs_vec_file, Dir)

   obs_sigma = Constant(obs_sigma) 
   reg = Constant(reg) 
   J = 0
   d_old = Constant(d_ini)
   for n in range(num_steps):
       d = forwardWellTransFunc(d_old, m, V, simul_params ) 
       d_old = d 
       d_obs = projectTstepHDF( mpi_comm, n, V, d_out_file, Dir )
       J += assemble( 0.5 * 1/(obs_sigma**2) * inner(obs_vec*(d- d_obs), obs_vec*(d- d_obs)) *dx ) 
   J /= num_steps
   J += assemble( 0.5 * 1/reg * inner( m- m_prior, m-m_prior)  * dx ) 
   
   control = Control(m) 
   reduced_functional = ReducedFunctional(J, control)   
   
   m_opt = minimize(reduced_functional, bounds = (m_lb, m_ub), options = {"disp":False}, tol=1e-6)
   writeVector(mpi_comm, V, m_opt, m_prior_file, Dir)

   return m_prior_file + '_out'

def writeVector(mpi_comm, V, m_opt, m_prior_file, Dir):
   x = V.tabulate_dof_coordinates()[:, 0:1] 
   y = V.tabulate_dof_coordinates()[:, 1:2] 
   m_out = m_opt.vector().get_local()[:, np.newaxis]
   output = np.concatenate((x, y, m_out), axis = 1) 
   mpi_rank = MPI.rank(mpi_comm) 
   name = Dir + m_prior_file + '_out' + str(mpi_rank) + '.txt'
   np.savetxt(name, output) 
   return name

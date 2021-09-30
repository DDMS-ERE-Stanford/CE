###########################################################
## forward.py: utility functions for forward simulation ###
###########################################################

import matplotlib.pyplot as plt
import numpy as np
from dolfin import *
from dolfin_adjoint import * 
import sys
import os 
from time import time 
from util.misc import *
TIME_SCALING_FACTOR = 10000 #time scaling factor to prevent too small coefficients  on simulation
tol = 1e-14
 
def boundary_r(x, on_boundary, R): 
    return on_boundary and (near(x[0], R, tol))

def boundary_l(x, on_boundary, L): 
    return on_boundary and (near(x[0], L, tol))

def boundary_t(x, on_boundary): 
    return on_boundary and (near(x[1], 1, tol))

def boundary_b(x, on_boundary): 
    return on_boundary and (near(x[1], 0, tol))

def boundary_well_1(x, on_boundary):
    tol = 0.01 
    return near(x[0], 20/81,  tol) and near(x[1], 1/81,  tol)

def boundary_well_2(x, on_boundary):
    tol = 0.01 
    return near(x[0], 20/81,  tol) and near(x[1], 30/81,  tol)

def boundary_well_3(x, on_boundary):
    tol = 0.01 
    return near(x[0], 20/81,  tol) and near(x[1], 53/81,  tol)

def boundary_well_4(x, on_boundary):
    tol = 0.01 
    return near(x[0], 20/81,  tol) and near(x[1], 75/81,  tol)


def boundary_well(x, on_boundary, n_wells, well_pos):
    tol = 20 
    b = False
    for i in range (n_wells):
        b = b or (near(x[0], well_pos[i][0], tol) and near(x[1], well_pos[i][1], tol))
    return b


def K(Y): 
    return pow(10,Y)*TIME_SCALING_FACTOR

def S(Y): 
    return 0.01*Y+0.1

def projectFunction(V, vec):
    Y = Function(V) 
    ordering = dof_to_vertex_map(V) 
    Y.vector()[:] = vec.flatten(order = 'C')[ordering]
    return Y    

def forwardPrevFunc( u_old, Y, V, simul_params): 
    lbc = simul_params["lbc"]
    rbc = simul_params["rbc"]
    dt = simul_params["dt_p"]/TIME_SCALING_FACTOR
    # bc 
    u_l = Constant(lbc) 
    u_r = Constant(rbc)
    del_t = Constant(dt)
    bc_l = DirichletBC(V, u_l, lambda x, on_boundary: boundary_l(x, on_boundary, 0)) 
    bc_r = DirichletBC(V, u_r, lambda x, on_boundary: boundary_r(x, on_boundary, simul_params["ngx"] * simul_params["dx"]))
    bcs = [bc_l, bc_r] 
    # variational form 
    u = TrialFunction(V) 
    v = TestFunction(V) 
    F = S(Y) * u * v * dx - S(Y) * u_old * v * dx + del_t * dot( K(Y) * grad(u), grad(v)) * dx
    a, L = lhs(F), rhs(F) 
    # solve PDE 
    u = Function(V) 
    solve( a == L, u, bcs)
    return u 
     
def forwardWellFunc(u_old, Y, V, simul_params): 
    set_log_level(LogLevel.ERROR)
    # get parameters 
    lbc = simul_params["lbc"]
    rbc = simul_params["rbc"]
    wCond  = simul_params["wCond"] 
    dt = simul_params["dt_w"]/TIME_SCALING_FACTOR
  #  dt = 0.001
    # boundary conditions 
    u_l = Constant(lbc)
    u_r = Constant(rbc)
    u_w = Constant(wCond)
    del_t = Constant(dt)
    bc_l = DirichletBC(V, u_l, lambda x, on_boundary: boundary_l(x, on_boundary, 0)) 
    bc_r = DirichletBC(V, u_r, lambda x, on_boundary: boundary_r(x, on_boundary, simul_params["ngx"] * simul_params["dx"]))
    bc_w = DirichletBC(V, u_w, lambda x, on_boundary: boundary_well(x, on_boundary, simul_params["n_wells"],\
                       simul_params["well_pos"]))
    bcs = [bc_l, bc_r, bc_w]
    # variational form 
    u = TrialFunction(V)
    v = TestFunction(V)
    F = S(Y) * u * v * dx - S(Y) * u_old * v * dx + del_t * dot( K(Y) * grad(u), grad(v)) * dx
    a, L = lhs(F), rhs(F)
    A, b = assemble_system(a, L, bcs)
    
    # solve PDE 
    u = Function(V)
    solve(A, u.vector(), b)
    return u


def forwardWell(perm, channel_params, simul_params): 
    perm = perm * (channel_params["hperm"] - channel_params["lperm"]) + channel_params["lperm"]
    ngx = simul_params["ngx"]
    ngy = simul_params["ngy"]
    dx = simul_params["dx"] 
    dy = simul_params["dy"]
    num_psteps = simul_params["num_psteps"] 
    num_wsteps = simul_params["num_wsteps"]
    
    u_ini = simul_params["ini_cond"] * np.ones((ngx*ngy, 1))
    u_sol = np.zeros((ngx*ngy, num_wsteps + 1))
    mesh = RectangleMesh(Point(0.0, 0.0), Point(ngx*dy, ngy*dy), ngy - 1, ngx - 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    
    Y = projectFunction(V, perm)
    u_prev = projectFunction(V, u_ini) 

    # prewell simulation 
    for n in range(num_psteps): 
        u = forwardPrevFunc( u_prev, Y, V, simul_params) 
        u_prev = u 
    u_sol[:,0:1] = u_prev.compute_vertex_values(mesh).reshape(ngy*ngx, 1)

    # after well installation 
    u_old = u_prev 
    for n in range(num_wsteps): 
        u = forwardWellFunc( u_old, Y, V, simul_params) 
        u_old = u
        u_sol[:,n+1:n+2] = u_old.compute_vertex_values(mesh).reshape(ngy*ngx, 1) 
    return u_sol

   

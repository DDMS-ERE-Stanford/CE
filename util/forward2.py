import matplotlib.pyplot as plt
import numpy as np
from dolfin import *
from dolfin_adjoint import * 
import sys
import os 
from time import time 
from util.misc import *
 
tol = 1e-14 
def boundary_r(x, on_boundary): 
    return on_boundary and (near(x[0], 1, tol))

def boundary_l(x, on_boundary): 
    return on_boundary and (near(x[0], 0, tol))

def boundary_t(x, on_boundary): 
    return on_boundary and (near(x[1], 1, tol))

def boundary_b(x, on_boundary): 
    return on_boundary and (near(x[1], 0, tol))

def boundary_well_1(x, on_boundary):
    tol = 0.01 
    return near(x[0], 0.5,  tol) and near(x[1], 0.3,  tol)

def boundary_well_2(x, on_boundary):
    tol = 0.01 
    return near(x[0], 0.5,  tol) and near(x[1], 0.7,  tol)


def boundary_well(x, on_boundary): 
    tol = 0.01 
    return near(x[0], 0.5,  tol) and near(x[1], 0.5,  tol)


def K(Y):  # log permeability
    return exp(Y)

def S(Y): # storage coefficient 
    return 0.1+0.05*Y

def forwardWellFunc( Y, V, simul_params): 
    set_log_level(LogLevel.ERROR)
    # get parameters 
    ngy = simul_params["ngy"] 
    ngx = simul_params["ngx"]
    lbc = simul_params["lbc"]
    rbc = simul_params["rbc"]
    wCond  = simul_params["wCond"] 
    # boundary conditions 
    u_l = Constant(lbc)
    u_r = Constant(rbc)
    u_w1 = Constant(0.2)
    u_w2 = Constant(0.2)
    bc_l = DirichletBC(V, lbc, boundary_l ) 
    bc_r = DirichletBC(V, rbc, boundary_r ) 
    bc_w1 = DirichletBC(V, u_w1, boundary_well_1, "pointwise")
    bc_w2 = DirichletBC(V, u_w2, boundary_well_2, "pointwise")
  #  bcs = [bc_l, bc_r]
    bcs = [bc_l, bc_r, bc_w1, bc_w2]
    # variational form 
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot( K(Y) * grad(u), grad(v)) * dx
    f = Constant(0.0) 
    L = f * v * dx 
    # solve PDE 
    u = Function(V)
    solve(a == L, u, bcs)
    return u 

def projectFunction(V, vec):
    Y = Function(V) 
    ordering = dof_to_vertex_map(V) 
    Y.vector()[:] = vec.flatten(order = 'C')[ordering]
    return Y    

def forwardWellTransFunc(u_old, Y, V, simul_params): 
    set_log_level(LogLevel.ERROR)
    # get parameters 
    lbc = simul_params["lbc"]
    rbc = simul_params["rbc"]
    wCond  = simul_params["wCond"] 
    dt = simul_params["dt"]
    # boundary conditions 
    u_l = Constant(lbc)
    u_r = Constant(rbc)
    u_w = Constant(wCond)
    del_t = Constant(dt)
    bc_l = DirichletBC(V, u_l, boundary_l) 
    bc_r = DirichletBC(V, u_r, boundary_r)
    bc_w = DirichletBC(V, u_w, boundary_well, "pointwise")
 #   bc_w1 = DirichletBC(V, u_w, boundary_well_1, "pointwise") 
 #   bc_w1 = DirichletBC(V, u_w, boundary_well_1, "pointwise") 
 #   bc_w2 = DirichletBC(V, u_w, boundary_well_2, "pointwise") 
    bcs = [bc_l, bc_r, bc_w]
  #  bcs = [bc_l, bc_r]
    # variational form 
    u = TrialFunction(V)
    v = TestFunction(V)
    F = S(Y) * u * v * dx - S(Y) * u_old * v * dx + del_t * dot( K(Y) * grad(u), grad(v)) * dx
    a, L = lhs(F), rhs(F)
    # solve PDE 
    u = Function(V)
    solve(a == L, u, bcs)
    return u

def forwardWellTransFunc2(u_old, Y, V, simul_params): 
    set_log_level(LogLevel.ERROR)
    # get parameters 
    lbc = simul_params["lbc"]
    rbc = simul_params["rbc"]
    wCond  = simul_params["wCond"] 
    dt = simul_params["dt"]
    # boundary conditions 
    u_l = Constant(lbc)
    u_r = Constant(rbc)
    u_w = Constant(wCond)
    del_t = Constant(dt)
    bc_l = DirichletBC(V, u_l, boundary_l) 
    bc_r = DirichletBC(V, u_r, boundary_r)
    bc_w = DirichletBC(V, u_w, boundary_well, "pointwise") 
    bcs = [bc_l, bc_r, bc_w]
    # variational form 
    u = TrialFunction(V)
    v = TestFunction(V)
    F = S(Y) * u * v * dx - S(Y) * u_old * v * dx + del_t * dot( K(Y) * grad(u), grad(v)) * dx 
    a, L = lhs(F), rhs(F)
    # solve PDE 
    u = Function(V)
    solve(a == L, u, bcs)
    return u


def forwardWellTrans(perm, simul_params): 
    ngx = simul_params["ngx"] # number of grids in x-direction
    ngy = simul_params["ngy"] # number of grids in y-direction
    dt = simul_params["dt"]  # time step size 
    num_steps = simul_params["num_steps"] # number of time steps 
    u_ini = simul_params["ini_cond"] * np.ones((ngx*ngy, 1)) #initial condition 
    u_sol = np.zeros((ngx*ngy, num_steps + 1))
    mesh = UnitSquareMesh(ngy - 1, ngx - 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    
    Y = projectFunction(V, perm)
    u_old = projectFunction(V, u_ini)
    u_sol[:,0:1] = u_ini
    
    for n in range(num_steps): 
        u = forwardWellTransFunc( u_old, Y, V, simul_params) 
        u_old = u
        u_sol[:,n+1:n+2] = u_old.compute_vertex_values(mesh).reshape(ngy*ngx, 1) 
    return u_sol



def forwardWellTrans2(perm, simul_params): 
    ngx = simul_params["ngx"]
    ngy = simul_params["ngy"]
    dt = simul_params["dt"] 
    num_steps = simul_params["num_steps"]
    u_ini = simul_params["ini_cond"] * np.ones((ngx*ngy, 1))
    u_sol = np.zeros((ngx*ngy, num_steps + 1))
    mesh = UnitSquareMesh(ngy - 1, ngx - 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    
    Y = projectFunction(V, perm)
    u_old = projectFunction(V, u_ini)
    u_sol[:,0:1] = u_ini
    
    for n in range(num_steps): 
        u = forwardWellTransFunc2( u_old, Y, V, simul_params) 
        u_old = u
        u_sol[:,n+1:n+2] = u_old.compute_vertex_values(mesh).reshape(ngy*ngx, 1) 
    return u_sol

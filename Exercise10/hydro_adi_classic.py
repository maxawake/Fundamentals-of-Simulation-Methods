import numpy as np
from bc import *

def advect(q,v,dx,dt):
    flux        = np.zeros_like(v)
    ipos        = np.where(v>=0.)[0]
    ineg        = np.where(v<0.)[0]
    flux[ipos]  = q[ipos]*v[ipos]
    flux[ineg]  = q[ineg+1]*v[ineg]
    qnew        = q.copy()
    qnew[1:-1] -= dt * ( flux[1:] - flux[:-1] ) / dx
    return qnew

def hydro_adi_classic_one_timestep(q,dx,dt,gamma=(7./5.)):
    rho    = q[0,:]
    u      = q[1,:]/rho
    uint   = 0.5 * ( u[1:] + u[:-1] )
    q[0,:] = advect(q[0,:],uint,dx,dt)
    q[1,:] = advect(q[1,:],uint,dx,dt)
    q[2,:] = advect(q[2,:],uint,dx,dt)
    rho    = q[0,:]
    u      = q[1,:]/rho
    etot   = q[2,:]/rho
    ekin   = 0.5*(u**2)
    eth    = etot-ekin
    p      = rho*eth*(gamma-1.)
    cs     = np.sqrt(gamma*p/rho)
    p      = rho*cs**2
    q[1,1:-1] += - dt * ( p[2:] - p[:-2] ) / ( 2*dx )
    q[2,1:-1] += - dt * ( p[2:]*u[2:] - p[:-2]*u[:-2] ) / ( 2*dx )
    return q

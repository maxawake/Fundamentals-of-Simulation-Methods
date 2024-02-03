import numpy as np
from hll import *

def hydro_adi_riemann_one_timestep(q,dx,dt,gamma=(7./5.)):
    rho        = q[0,:]
    u          = q[1,:]/rho
    etot       = q[2,:]/rho
    ekin       = 0.5*(u**2)
    eth        = etot-ekin
    p          = rho*eth*(gamma-1.)
    cs         = np.sqrt(gamma*p/rho)
    f          = np.zeros_like(q)
    f[0,:]     = rho*u
    f[1,:]     = rho*u**2 + p
    f[2,:]     = ( rho*etot + p ) * u
    fint       = hll_interface_flux(q,f,dx,dt,gamma=gamma)
    q[:,1:-1] -= (dt/dx)*(fint[:,1:]-fint[:,:-1])
    return q

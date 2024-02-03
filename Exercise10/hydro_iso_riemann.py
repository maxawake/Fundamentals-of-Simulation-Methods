import numpy as np
from hll import *

def hydro_iso_riemann_one_timestep(q,cs,dx,dt):
    rho        = q[0,:]
    u          = q[1,:]/rho
    p          = rho*cs**2
    f          = np.zeros_like(q)
    f[0,:]     = rho*u
    f[1,:]     = rho*u**2 + p
    fint       = hll_interface_flux(q,f,dx,dt,cs=cs)
    q[:,1:-1] -= (dt/dx)*(fint[:,1:]-fint[:,:-1])
    return q

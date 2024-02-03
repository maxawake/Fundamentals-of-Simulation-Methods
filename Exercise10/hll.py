import numpy as np

def hll_interface_flux(q,f,dx,dt,cs=None,gamma=None):
    rho        = q[0,:]
    u          = q[1,:]/rho
    if cs is None:
        if gamma is None:
            exit("Must either specify cs (isothermal hydro) or gamma (adiabatic hydro)")
        etot   = q[2,:]/rho
        ekin   = 0.5*(u**2)
        eth    = etot-ekin
        p      = rho*eth*(gamma-1.)
        cs     = np.sqrt(gamma*p/rho)
    lamneg     = u[:-1] - cs[:-1]
    lampos     = u[1:]  + cs[1:]
    fint       = (lampos*f[:,:-1]-lamneg*f[:,1:]+lampos*lamneg*(q[:,1:]-q[:,:-1]))/(lampos-lamneg)
    ii         = np.where(lamneg>0)[0]
    fint[:,ii] = f[:,ii]
    ii         = np.where(lampos<0)[0]
    fint[:,ii] = f[:,ii+1]
    return fint

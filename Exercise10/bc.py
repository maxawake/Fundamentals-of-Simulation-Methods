def implement_boundcond(q,bctype='periodic'):
    if bctype == 'periodic':
        q[:,0]   = q[:,-2]
        q[:,-1]  = q[:,1]
    if bctype == 'fixed':
        q[:,0]   = q[:,1]
        q[:,-1]  = q[:,-2]

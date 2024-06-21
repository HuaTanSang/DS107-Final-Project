import numpy as np 

def NoisyStat(x, y, xm, ym, n, eps, xnew):
    delta = (1 - 1/n)
    eps = eps/3.0

    l1 = np.random.laplace(0., 3 * delta/eps, 1)
    l2 = np.random.laplace(0., 3 * delta/eps, 1)

    nvar_x = np.dot(x-xm, x-xm)
    nvar_xy = np.dot(x-xm, y-ym)

    if nvar_x + l2 <= 0: 
        return None 
    
    alpha = (nvar_xy + l1) / (nvar_x + l2) 
    delta_3 = (1 /n) * (1 + np.abs(alpha))

    l3 = np.random.laplace(0, 3 * delta_3 / eps)
    beta = (y.mean() - alpha * xm) + l3 
    
    return alpha * xnew + beta



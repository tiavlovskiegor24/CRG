import numpy as np

def GMFPT_theory(d,l,v,weighted=True,select = "all") :
    """
    According to the theory of Lin et al., 2012, the global mean first passage
    time can be calculated by finding the eigenspectrum of the Laplacian matrix
    of the graph. This function calculates the GMFPT from their formula, for the
    graph described by the adjacency matrix A, to all sites. Optional parameter
    'weighted' allows for the choice of having the same quantity but weighted
    with the stationary distribution.
    
    arguments:
        d - degree vector
        l - eigenvalues
        v - eigenvectors
        select - which eigenvectors to select
    """
    
    N = d.shape[0]
    E = np.sum(d)/2.
    
    sortidx = np.argsort(l)
    if select != 'all':
        sortidx = sortidx[select]
        print sortidx.shape
    l = l[sortidx]
    v = v[:,sortidx]
    T = np.zeros(N)
    dv = np.dot (v.T,d)

    if not weighted :
        T = np.dot(2.0*E*v[:,1:]**2 - v[:,1:]*dv[1:],l[1:]**-1)
        return float(N)/(N-1.0) * T
    else :
        T = np.dot((2.0*E)**2*v[:,1:]**2 - 2*(2*E)*v[:,1:]**2*dv[1:]\
                   - (v[:,1:]*dv[1:])**2,l[1:]**-1)
        return T/(2.*E)
# extract anti-diagonals of sparse matrix 
import numpy as np



def anti_diags(H,k = None,sym = True):
    n = H.shape[0]
    adiags = []
    for i in xrange(n):
        adiags.append(H[(np.arange(i),np.arange(i,-1,-1))])
        

    return adiags

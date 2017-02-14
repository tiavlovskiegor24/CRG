import numpy as np
import matplotlib.pyplot as plt
from myplot import myplot

def gen_dec_feature_vec(M,dbins_max = 100):
    n = M.shape[0]
    dec_vec = np.zeros(n)
    for loc,data in enumerate(M):
        dbins = np.arange(n)- loc
        data = data.toarray().ravel()
        if loc > 0:
            try:
                
                data[loc+1:2*loc+1] = np.maximum(data[:loc],\
                                             data[loc+1:2*loc+1])
            except:
                print loc

        data = data[loc+1:]
        dbins = dbins[loc+1:]

        data = data*1./data.sum()# probability distribution
        #data = np.cumsum(pdata[::-1])[::-1]#cum pd from the tail
        data = data[:dbins_max]
        dbins = dbins[:dbins_max]
        
        dec_vec[loc] = compute_decay(data,dbins)[0]
        
    return dec_vec
    

def compute_decay(contacts,dbins = None,plot=False):
    '''
    function takes raw hic matrix row and computes a power law decay rate of contacts with respect to separation in bins
    '''

    # copute contact probabilities
    #p = contacts*1./contacts.sum()
    #print p
    
    if dbins is None:
        dbins = np.arange(1,contacts.shape[0]+1)

    
    fit = np.polyfit(np.log10(dbins),np.log10(contacts),1)
    fit_fn = np.poly1d(fit) 

    if plot:
        f,ax = myplot(np.log10(contacts),np.log10(dbins),style = 'yo')
        ax.plot(np.log10(dbins), fit_fn(np.log10(dbins)), '--k')

    return fit
    
    

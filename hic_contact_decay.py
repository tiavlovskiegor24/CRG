import numpy as np
import matplotlib.pyplot as plt
from myplot import myplot

def process_data(data):
    data = (data+1)*1./(data+1).sum()# probability distribution using\
               #Laplace rule
    #data = np.cumsum(data[::-1])[::-1]#cum pd from the tail
    return data

def gen_dec_feature_vec(M,dbins_max = 100):
    '''
    
    '''
    
    n = M.shape[0]
    dec_vec = np.zeros(n)
    dbins = np.arange(1,dbins_max+1)
    
    data = M[0].toarray().ravel()[1:dbins_max+1]
    data = process_data(data)
    dec_vec[0] = compute_decay(data,dbins,plot = True)[0]
    plt.title("bin %d, decay rate = %.2f"%(0,dec_vec[0]))

    if n < 2:
        return dec_vec
    
    data = M[n-1].toarray().ravel()[::-1][1:dbins_max+1]
    data = process_data(data)   
    dec_vec[n-1] = compute_decay(data,dbins,plot = True)[0]
    plt.title("bin %d, decay rate = %.2f"%(n-1,dec_vec[n-1]))

    sample = np.sort(np.random.randint(1,n,5)).tolist()[::-1]

    for loc in xrange(1,n/2):
        #print "First",loc
        data = M[loc].toarray().ravel()
        data[loc+1:2*loc+1] = np.maximum(data[:loc][::-1],\
                                         data[loc+1:2*loc+1])
        data = data[loc+1:]

        data = process_data(data)

        data = data[:dbins_max]

        if sample:
            if loc == sample[-1]:
                 dec_vec[loc] = compute_decay(data,dbins,plot=True)[0]
                 plt.title("bin %d, decay rate = %.2f"%(loc,dec_vec[loc]))
                 sample.pop()
                 continue
        
        dec_vec[loc] = compute_decay(data,dbins)[0]
            
    for loc in xrange(n/2,n-1):
        #print "Second",loc
        data = M[loc].toarray().ravel()
        data[loc-(n-loc-1):loc] = np.maximum(data[loc-(n-loc-1):loc],\
                                             data[loc+1:][::-1])

        data = data[:loc][::-1]

        data = process_data(data)


        data = data[:dbins_max]

        if sample:
            if loc == sample[-1]:
                 dec_vec[loc] = compute_decay(data,dbins,plot=True)[0]
                 sample.pop()
                 plt.title("bin %d, decay rate = %.2f"%(loc,dec_vec[loc]))
                 continue
        
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
    
    

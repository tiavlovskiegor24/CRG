# plot vector and its outer product matrix
import matplotlib.pyplot as plt
import numpy as np

def plot_outer(ab):

    f,ax = plt.subplots(1,2,sharey=True, figsize = (8,6),\
                        gridspec_kw = {'width_ratios':[1, 3]})
    ax1,ax2 = ax

    #ax1.plot(np.diff(ab),np.arange(ab.shape[0]-1))
    ax1.plot(ab,np.arange(ab.shape[0]))
    ax1.vlines(ab.mean(),0,ab.shape[0],"r")


    im = ax2.matshow (np.log1p(np.outer(ab,ab)),cmap=plt.cm.seismic,\
                      origin='upper',interpolation='none')
    ax2.set_aspect("equal")
    #f.colorbar(im)

    f.tight_layout()

    return f,ax

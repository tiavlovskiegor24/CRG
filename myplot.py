import matplotlib.pyplot as plt
import numpy as np

def myplot(indices = None,values = None,style = "b",shape = (1,1),figsize = (14,4),**kwargs):

    f,ax = plt.subplots(shape[0],shape[1],figsize = (figsize[0],figsize[1]*shape[0]),**kwargs)

    if indices is None:
        return f,ax
        
    if values is None:
        values = indices
        indices = np.arange(len(values))
        
    try:
        ax.plot(indices,values,style,**kwargs)
        ax.set_xlim(xmax = max(indices),xmin = min(indices))
    except Exception as e:
        #print e
        ax[0].plot(indices,values,style)
        ax[0].set_xlim(xmax = max(indices),xmin = min(indices))
        
    f.tight_layout()
    return f,ax

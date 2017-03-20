import matplotlib.pyplot as plt
import numpy as np

def myplot(values = None,indices = None,style = "b",shape = (1,1),**kwargs):
    f,ax = plt.subplots(shape[0],shape[1],figsize = (14,4*shape[0]),**kwargs)
    if values is None:
        return f,ax
    if indices is None:
        indices = np.arange(len(values))
    ax.plot(indices,values,style)
    ax.set_xlim(xmax = max(indices),xmin = min(indices)) 
    f.tight_layout()
    return f,ax

import matplotlib.pyplot as plt
import numpy as np

def myplot(values = None,indices = None,style = "_",shape = (1,1)):
    f,ax = plt.subplots(shape[0],shape[1],figsize = (10,4))
    if values is None:
        return f,ax
    if indices is None:
        indices = np.arange(len(values))
    ax.plot(indices,values,style)
    ax.set_xlim(xmax = max(indices)) 
    f.tight_layout()
    return f,ax

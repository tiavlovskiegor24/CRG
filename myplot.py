import matplotlib.pyplot as plt

def myplot(array,shape = (1,1)):
    f,ax = plt.subplots(shape[0],shape[1],figsize = (10,4))
    ax.plot(array)
    ax.set_xlim(xmax = len(array))
    f.tight_layout()
    return f,ax

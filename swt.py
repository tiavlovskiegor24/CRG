# wavelet analysis of eigenvector
import matplotlib.pyplot as plt
import numpy as np
import pywt

def swt(ev,level = "max",wavelet = "sym4",plot = True,rec = True):
    # compute stationary wavelet coefficients of series ev down
    # to level and plot the reconstructed
    # smooth and details at each level
    # return smooth and details if rec is true
    # coefficients at each level otherwise
    
    coefs= pywt.swt(ev.ravel(),wavelet)
    nlevels = len(coefs)
    if level == "max":
        level = nlevels-1
    
    coefs1 = [(np.zeros_like(ev.ravel()),np.zeros_like(ev.ravel())) \
              for i in xrange(len(coefs))]
    coefs1[:nlevels-level] = coefs[:nlevels-level]

    if rec:
        smooth = np.array(pywt.iswt(coefs1,wavelet))
    else:
        smooth = np.array(coefs1[0][0])

    if plot:
        f,ax1 = plt.subplots(1,1,sharex=True,figsize=(10,6))
        #ax1,ax2 = ax

        ax1.plot(ev,alpha = 0.5)
        ax1.plot(smooth,"r")
        ax1.set_xlim(xmax = len(smooth))
        ax1.set_yticks([])

        offset = 0
        details = np.zeros((level,len(smooth)))
        for l in (np.arange(level)+1)[::-1]:
            coefs2 = [(np.zeros_like(ev.ravel()),np.zeros_like(ev.ravel()))\
                      for i in xrange(len(coefs))]
            coefs2[nlevels-l] = (coefs2[nlevels-l][0],coefs[nlevels-l][1])
            if rec:
                details[-l,:] = np.array(pywt.iswt(coefs2,wavelet))
            else:
                details[-l,:] = coefs2[nlevels-l][1]
                
            offset += np.max(np.abs(details[-l,:]))
            #smooth += detail
            ax1.plot(details[-l,:]-offset,"b")
            offset += np.max(np.abs(details[-l,:]))

        #ax1.plot(smooth,"g",alpha = .6)
        f.tight_layout()
    
    return smooth,details

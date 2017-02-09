# maximum overlap wavelet transform  of a signal 
import matplotlib.pyplot as plt
import numpy as np
import pywt

class swt(object):

    def __init__(self,ev,level = "max",wavelet = "sym4",plot = True,rec = True):
        
               
        if level == "max":
            self.slices = self.split_idx(ev)
            self.nlevels = pywt.swt_max_level(len(ev[self.slices[0]]))
            level = self.nlevels-1


        else:
            self.slices = self.split_idx(ev)
            self.nlevels = pywt.swt_max_level(len(ev[self.slices[0]]))
            #level = self.nlevels-1


        self.smooth = np.zeros_like(ev)
        self.details = np.zeros((level,ev.shape[0]))

        for s in self.slices[:3]:

            smooth,details = self.comp_coeffs(ev[s],level,wavelet = "sym4",rec = True)
            self.smooth[s] = self.smooth[s] + smooth[:]
            self.details[:,s] = self.details[:,s] + details[:,:]
            #self.swt_plot(level,ev[s],smooth,details=None)


        self.smooth[self.slices[2]] = self.smooth[self.slices[2]]/2. 
        self.smooth[self.slices[3]] = self.smooth[self.slices[3]]*2./3
        #self.smooth[self.slices[3]] = smooth[self.slices[2]]

        self.details[:,self.slices[2]] = self.details[:,self.slices[2]]/2. 
        self.details[:,self.slices[3]] = self.details[:,self.slices[3]]*2./3

        if plot:
            self.swt_plot(level,ev,self.smooth,self.details)
    
    
    def split_idx(self,ev,max_scale = None):

        if max_scale > np.log2(ev.shape[0]) or max_scale is None:
            max_scale = int(np.log2(ev.shape[0]))

        mod = ev.shape[0] % 2**(max_scale)
        s = []
        s.append(np.s_[mod:])
        s.append(np.s_[:-mod])
        s.append(np.s_[mod/2:-(mod-mod/2)])
        s.append(np.s_[mod:-mod])
        return s

    def swt_plot(self,level,ev = None,smooth=None,details=None):
        # function plots swt smooth and details
        
        f,ax1 = plt.subplots(1,1,sharex=True,figsize=(10,6))
        #ax1,ax2 = ax

        if ev is not None:
            ax1.plot(ev,alpha = 0.5)
            #ax1.set_ylim(ymax = ev.max()*1.2)
            ax1.set_xlim(xmax = len(ev))

        if smooth is not None: 
            ax1.plot(smooth,"r")
            ax1.set_xlim(xmax = len(smooth))

        ax1.set_yticks([])

        if details is not None: 
            offset = 0
            for l in (np.arange(level)+1)[::-1][:]:
                skip = np.max(np.abs(details[-l,:]))
                offset += skip
                ax1.plot(details[-l,:]-offset,"b")
                offset += skip
            ax1.set_xlim(xmax = details.shape[1])
        f.tight_layout()

    def comp_coeffs(self,ev,level = "max",wavelet = "sym4",plot = True,rec = True):
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

        details = np.zeros((level,len(smooth)))

        for l in (np.arange(level)+1)[::-1]:
            coefs2 = [(np.zeros_like(ev.ravel()),np.zeros_like(ev.ravel()))\
                          for i in xrange(len(coefs))]
            coefs2[nlevels-l] = (coefs2[nlevels-l][0],coefs[nlevels-l][1])
            if rec:
                details[-l,:] = np.array(pywt.iswt(coefs2,wavelet))
            else:
                details[-l,:] = np.array(coefs2[nlevels-l][1])


        return smooth,details

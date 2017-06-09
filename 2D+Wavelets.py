
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')


# In[2]:

import pywt
#import cooler

from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import matplotlib

import numpy as np


# In[690]:

array = np.loadtxt("forEgor.txt")


# In[742]:

non_mappable = array.sum(axis = 0) < 100


# In[743]:

array  = array[~non_mappable,:][:,~non_mappable]


# In[744]:

f = plt.figure(figsize=(10,10))
plt.imshow(-np.log1p(array), cmap='gray')


# In[683]:

f = plt.figure(figsize=(10,10))
plt.imshow(-np.log1p(array), cmap='gray')


# In[636]:

myplot(np.log1p(array[:,50]))


# In[633]:

f = plt.figure(figsize=(10,10))
plt.matshow(-np.log1p(array),fignum=1,cmap = "gray")


# In[796]:

wavelet = "haar"
coeffs = pywt.wavedec2(-np.log1p(array),wavelet=wavelet)


# In[782]:

len(coeffs)


# In[783]:

np.sum(coeffs[5][0]**2,axis=1)


# In[798]:

f = plt.figure(figsize=(10,10))
plt.matshow(coeffs[5][2],fignum=f.number,cmap = "gray")


# In[654]:

recoeffs = [np.zeros_like(coeffs[0])]+[level if i+1 in {5,6} else (np.zeros_like(level[0]),None,None) for i,level in enumerate(coeffs[1:])]


# In[105]:

recoeffs = [np.zeros_like(coeffs[0])]+        [(level[0],None,None) if i in {6} else tuple(np.zeros_like(level[0])         for _ in range(3)) for i,level in enumerate(coeffs[1:])]


# In[187]:

recoeffs = [None]+[(level[0],None,None) if i+1 in {8} else tuple(np.zeros_like(level[0])                                                                     for _ in range(3)) for i,level in enumerate(coeffs[1:])]


# {5,6,7} for haar

# In[792]:

#vertical details
levels = {5,6,7}
recoeffs = [None]+[(level[0],None,None) if i+1 in levels else (np.zeros_like(level[0]),None,None) for i,level in enumerate(coeffs[1:])]


# In[564]:

# horizontal and vertical details
levels = {5}
recoeffs = [None]+[(level[0],level[1],None) if i+1 in levels else (np.zeros_like(level[0]),None,None) for i,level in enumerate(coeffs[1:])]


# In[802]:

# vertical and diagonal details
levels = {5,6,7,8}
recoeffs = [None]+[(level[0],None,level[2]) if i+1 in levels else (np.zeros_like(level[0]),None,None) for i,level in enumerate(coeffs[1:])]


# In[566]:

# horizontal,vertical and diagonal details
levels = {5}
recoeffs = [None]+[level if i+1 in levels else (np.zeros_like(level[0]),None,None) for i,level in enumerate(coeffs[1:])]


# In[555]:

# diagonal details
levels = {5,6}
recoeffs = [None]+[(None,None,level[2]) if i+1 in levels else (np.zeros_like(level[0]),None,None) for i,level in enumerate(coeffs[1:])]


# In[801]:

recmat = pywt.waverec2(recoeffs,wavelet=wavelet)

f,ax = plt.subplots(1,2,figsize=(24,12),sharey=True)
ax[0].matshow(recmat,cmap = "gray")
ax[0].set_aspect("equal")
ax[1].matshow(-np.log1p(array),cmap = "gray")
ax[1].set_aspect("equal")
f.tight_layout()


# In[803]:

recmat = pywt.waverec2(recoeffs,wavelet=wavelet)

f,ax = plt.subplots(1,2,figsize=(24,12),sharey=True)
ax[0].matshow(recmat,cmap = "gray")
ax[0].set_aspect("equal")
ax[1].matshow(-np.log1p(array),cmap = "gray")
ax[1].set_aspect("equal")
f.tight_layout()


# In[628]:

array = np.load("hic-2.0-2000-1.npy")


# In[788]:

col_1 = 350
col_2 = 390
f,ax = myplot(recmat[:,col_1]**2,figsize=(24,12),shape=(1,2),sharey = True)
ax[1].plot(recmat[:,col_2]**2)
#ax[0].set_ylim(ymax = 0.015,ymin = 0)
f.tight_layout()


# In[789]:

myplot(np.sum(recmat**2,axis=0))


# In[804]:

myplot(np.sum(recmat**2,axis=0))


# In[677]:

get_ipython().run_cell_magic(u'javascript', u'', u'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# In[790]:

fig = plt.figure(figsize=(10,14))
gs = plt.GridSpec(2,1,hspace = 0,height_ratios=[5,2])
ax1 = plt.subplot(gs[0,0])
ax1.matshow(-np.log1p(array),cmap = "gray")
ax = plt.subplot(gs[1,0],sharex=ax1)
ax.plot(np.sum(recmat**2,axis=0))


# In[805]:

fig = plt.figure(figsize=(10,14))
gs = plt.GridSpec(2,1,hspace = 0,height_ratios=[5,2])
ax1 = plt.subplot(gs[0,0])
ax1.matshow(-np.log1p(array),cmap = "gray")
ax = plt.subplot(gs[1,0],sharex=ax1)
ax.plot(np.sum(recmat**2,axis=0))


# In[736]:

col_1 = 750
col_2 = 400
f,ax = myplot(recmat[:,col_1]**2,figsize=(24,12),shape=(1,2),sharey = True)
ax[1].plot(recmat[:,col_2]**2)
f.tight_layout()


# In[735]:

col_1 = 750
col_2 = 580
f,ax = myplot(recmat[:,col_1]**2,figsize=(24,12),shape=(1,2),sharey = True)
ax[1].plot(recmat[:,col_2]**2)
f.tight_layout()


# In[733]:

np.sum(recmat[:,col_1]**2)


# In[734]:

np.sum(recmat[:,col_2]**2)


# In[500]:

col_1 = 370
col_2 = 400
f,ax = myplot(recmat[:,col_1]**2,figsize=(24,12),shape=(1,2),sharey = True)
ax[1].plot(recmat[:,col_2]**2)
f.tight_layout()


# In[475]:

myplot(recmat[:,400],figsize=(7,6))


# In[436]:

myplot(np.log1p(array[:,370]),figsize=(7,6))


# In[435]:

f,ax = myplot(np.log1p(array[:,370]),figsize=(16,6))
ax.plot(-recmat[:,370],"r")


# In[424]:

myplot(recmat[580,:],figsize=(7,6))


# In[279]:

myplot(recmat[:,],figsize=(7,6))


# In[326]:

np.sum(recmat[:,800]**2)


# In[327]:

np.sum(recmat[:,750]**2)


# In[325]:

myplot(recmat[:,800]**2,figsize=(7,6))


# In[324]:

myplot(recmat[:,750]**2,figsize=(7,6))


# In[276]:

myplot(recmat[500,:]**2,figsize=(7,6))


# In[ ]:




# In[275]:

myplot(np.sum(recmat**2,axis = 0),figsize=(7,6))


# In[256]:

get_ipython().magic(u'run ../scripts/myplot.py')


# In[95]:

f = plt.figure(figsize=(10,10))
plt.imshow(-np.log1p(array), cmap='gray')


# In[ ]:




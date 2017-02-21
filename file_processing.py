import numpy as np
import pandas as pd

def read_feature_file(filename,**kwargs):
    return np.loadtxt(filename, dtype={'names': ('chrom', 'bin_start', 'bin_end','value'),
                                       'formats': ('a10', 'i', 'i','f')},delimiter="\t",**kwargs)

def write_feature_file(filename,array,feature_fmt = "%.3f",**kwargs):
    chrom_fmt = "%"+str(len(max(array["chrom"],key=len)))+"s"
    fmt = chrom_fmt+"\t%d\t%d\t"+feature_fmt
    np.savetxt(filename,array,fmt = fmt,**kwargs)
    
    
    
    
    

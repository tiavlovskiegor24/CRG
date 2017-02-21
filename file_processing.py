import numpy as np
import pandas as pd

def read_feature_file(filename):
    return np.loadtxt(filename, dtype={'names': ('chrom', 'bin_start', 'bin_end','value'),
                                       'formats': ('a10', 'i', 'i','f')},delimiter="\t",skiprows=1)

def write_feature_file(filename):
    pass
    
    
    
    

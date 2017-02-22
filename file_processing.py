import numpy as np
import pandas as pd

def read_feature_file(filename,as_dict = True,fill = None, **kwargs):
    array = np.loadtxt(filename, dtype={'names': ('chrom', 'bin_start', 'bin_end','value'),\
                                        'formats': ('a10', 'i', 'i','f')},delimiter="\t",**kwargs)
    if as_dict:
        array_dict = {}
        for chrom in np.unique(array["chrom"]):
            array_dict[chrom] = array[np.where(array["chrom"]==chrom)]

        array = array_dict

    if fill in not None:
        array = fill_array(array,fill)
        
    return array

def write_feature_file(filename,array,feature_fmt = "%.3f",**kwargs):
    chrom_fmt = "%"+str(len(max(array["chrom"],key=len)))+"s"
    fmt = chrom_fmt+"\t%d\t%d\t"+feature_fmt
    np.savetxt(filename,array,fmt = fmt,**kwargs)
    
def fill_array(array,fill_value = None):
    if fill_value is None:
        fill_value = np.nan

    res = array["bin_end"][0]-array["bin_start"][0]

    chrom_sizes = get_chrom_sizes()
    
    full_array = np.array([],dtype={'names': ('chrom', 'bin_start', 'bin_end','value'),
                                       'formats': ('a10', 'i', 'i','f')})
    for chrom in np.unique(array['chrom']):
        bins = array["bin_start"]
        
        
    if array.shape[0] < array["bin_end"][-1]/res:
        n = max(array.shape[0],array["bin_end"][-1]/res)
        
        full_array["names"] = 
        
    else:
        return array

def get_chrom_sizes(dir = None):
    if dir is None:
        dir = "/mnt/shared/data/HiC_processing/hg19_chromsizes.txt"
    
    chroms,sizes = np.loadtxt(dir,dtype=np.str,unpack=True)

    return dict(zip(chroms,sizes.astype(np.int)))


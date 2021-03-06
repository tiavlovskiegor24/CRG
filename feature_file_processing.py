import numpy as np
import pandas as pd
    
def get_feature_file_format():
    return {'names': ('chrom', 'bin_start', 'bin_end','value'),\
            'formats': ('a16', 'i', 'i','f')}
    
def read_feature_file1(filename,as_dict = True,impute = None, **kwargs):
    array = np.loadtxt(filename, dtype = get_feature_file_format(),delimiter="\t",**kwargs)
    if as_dict:
        array_dict = {}
        for chrom in np.unique(array["chrom"]):
            array_dict[chrom] = array[np.where(array["chrom"]==chrom)]

        array = array_dict

    if impute is not None:
        array = full_array(array,impute)
        
    return array

def read_feature_file(filename,as_dict = True,impute = None, **kwargs):
    data = []
    with open(filename,"r") as f:
        for line in f:
            line = line.strip("\n").split("\t")
            try:
                chrom,bin_start,bin_end,value = line[0],int(line[1]),int(line[2]),float(line[3])
            except:
                print line
            data.append((chrom,bin_start,bin_end,value))

            
    array = np.array(data,dtype=get_feature_file_format())
    del data
    if as_dict:
        array_dict = {}
        for chrom in np.unique(array["chrom"]):
            array_dict[chrom] = array[np.where(array["chrom"]==chrom)]

        array = array_dict

    if impute is not None:
        array = full_array(array,impute)
        
    return array

def write_feature_file1(filename,data,res,feature_fmt = "%.3f",**kwargs):
    # data is a tuple of chromosome, bins and values vectors 
    
    chroms,bins,values = data

    n = len(values)

    
    if bins is None:
        bins = np.arange(n)
        
    if isinstance(chroms,str):
        chroms = [chroms for i in xrange(n)]
        
    array = np.array(zip(chroms,\
                         [b*res for b in bins],\
                         [(b+1)*res for b in bins],\
                         values),\
                     dtype=get_feature_file_format())

    

    
    chrom_fmt = "%"+str(len(max(array["chrom"],key=len)))+"s"
    fmt = chrom_fmt+"\t%d\t%d\t"+feature_fmt
    np.savetxt(filename,array,fmt = fmt,**kwargs)

def write_feature_file(f,data,res,**kwargs):
    #with open(filename,"a") as f:
    for chrom,b,value in data:
        f.write("{}\t{}\t{}\t{}\n".format(chrom,b*res,(b+1)*res,value))
    

def full_array(array,res = None,fill_value = np.nan):

    chrom_sizes = get_chrom_sizes()
    
    if isinstance(array,dict):
        for chrom in array:
            
            if res is None:
                res = array[chrom]["bin_end"][0]-array[chrom]["bin_start"][0]


            if array[chrom].shape[0] == array[chrom]["bin_start"][-1]/res+1:
                continue

            bins = array[chrom]["bin_start"]/res
            if chrom in chrom_sizes:
                n = chrom_sizes[chrom]/res+1
            else:
                continue

            f_array = np.array(zip([chrom for i in xrange(n)],\
                                   np.arange(n)*res,\
                                   np.arange(1,n+1)*res,\
                                   np.ones(n)*fill_value),\
                               dtype=array[chrom].dtype)

            f_array["value"][bins] = array[chrom]["value"]
            array[chrom] = f_array
    else:
        pass
        '''
        for chrom in np.unique(array['chrom']):
            bins = array["bin_start"]/res        
        '''
        
    return array

def ravel_feature_data(data):
    out = None
    for chrom in data:
        if out is not None:
            out = np.concatenate((out,data[chrom]),axis = 0)
        else:
            out = data[chrom]
    return out
        

def get_chrom_sizes(dir = None):
    if dir is None:
        dir = "/mnt/shared/data/HiC_processing/hg19_chromsizes.txt"
    
    chroms,sizes = np.loadtxt(dir,dtype=np.str,unpack=True)

    return dict(zip(chroms,sizes.astype(np.int)))


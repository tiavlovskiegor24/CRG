#function to extract hi-c matrix from cooler file format
import cooler
import numpy as np

def hi_c_cooler(filepath,chrom):

    # load the data
    c = cooler.Cooler(filepath)

    # select indivudual chromosomes
    cis = c.matrix().fetch(chrom).tocsr().astype(np.float)
    print "Matrix shape:",cis.shape
    print "Density %.2f"%(cis.nnz*1./cis.shape[0]**2)
    return cis

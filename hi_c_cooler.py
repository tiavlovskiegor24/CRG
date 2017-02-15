#function to extract hi-c matrix from cooler file format
import cooler
import numpy as np
from scipy import sparse

def hi_c_cooler(filepath,chrom):

    # load the data
    c = cooler.Cooler(filepath)

    # select indivudual chromosomes
    cis = sparse.csr_matrix(c.matrix(balance = False).fetch(chrom)).astype(np.float)
    print "Matrix shape:",cis.shape
    print "Density %.2f"%(cis.nnz*1./cis.shape[0]**2)
    return cis
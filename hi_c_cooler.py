#function to extract hi-c matrix from cooler file format
import cooler
import numpy as np
from scipy import sparse

def hi_c_cooler(filepath,chrom,res = None,bins=True):

    if res is None:
        print "Input the resolution"
        return
    # load the data
    c = cooler.Cooler(filepath)

    # select indivudual chromosomes
    cis = sparse.csr_matrix(c.matrix(balance = False).fetch(chrom)).astype(np.float)
    print "Matrix shape:",cis.shape
    print "Density %.2f"%(cis.nnz*1./cis.shape[0]**2)

    if bins:
        # create vector of bin indices
        bins = c.bins().fetch(chrom)
        bins = (bins["start"].values*1./res).astype(np.int)
    else:
        bins = None

    return cis,bins

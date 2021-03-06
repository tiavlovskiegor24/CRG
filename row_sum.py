import numpy as np
import matplotlib.pyplot as plt
from myplot import myplot
import cooler
from scipy import sparse
from scipy.linalg import eigh
from hi_c_cooler import hi_c_cooler
from sparsify import sparsify
from time import time
from myplot import myplot
from feature_file_processing import write_feature_file


def row_sum_feature(newfilename,filepath,res = None, chrom = None):
    
    
    if res is None:
        print "Enter resolution"
        return

    with open(newfilename,"wb") as f:

        c = cooler.Cooler(filepath)

        if chrom is None:
            chroms = [str(chrom) for chrom in c.chromnames]
        else:
            chroms = [chrom]

        for chrom in chroms:
            print "Processing %s"%chrom

            cis,bins = hi_c_cooler(c,chrom,res = res)
            n = cis.shape[0]
            
            row_sum = np.sum(cis,axis = 1).astype(np.int)
            
            myplot(bins,row_sum)
            plt.title(chrom)
            plt.show()
            '''
            out = np.c_[[chrom for i in xrange(n)],\
                  np.arange(n)*res,\
                  np.arange(1,n+1)*res,\
                        row_sum]
            out = out.astype(np.str)
            '''
            #np.savetxt(f,out,fmt = "%s",delimiter = "\t")

            write_feature_file(f,data = (chrom,bins,row_sum),res = res,feature_fmt = "%d")
            print "%s finished\n"%chrom

        print "Everything is finished" 

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
from file_processing import write_feature_file


def Compute_GMFPT_Feature(newfilename,filepath,res = None, chrom = None):
    
    
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

            cis,bins = hi_c_cooler(filepath,chrom,res = res)
            n = cis.shape[0]
            if n < 2:
                continue

            # Adjacency matrix
            A = cis - sparse.diags(cis.diagonal())
            del cis

            # remove 1 read entries
            A = sparsify(A,2)

            #remove non-reachable regions
            D = np.asarray((A>0).sum(axis=1)).ravel()
            select = np.where(D>0)[0]
            A = A[select,:][:,select]
            bins = bins[select]
            #del select

            # Construct
            # Degree vector
            D = np.asarray(A.sum(axis=1)).ravel()

            # Laplacian matrix
            L = sparse.diags(D) - A
            del A

            # normalise Laplacian
            #L = np.dot(np.dot(sparse.diags(D**(-1),0),L),sparse.diags(D**(-1),0))

            print "Computing Eigenvectors of Laplacian"
            t1 = time()
            evals, evecs = eigh(L.toarray())
            print "Time taken: %.3f"%(time()-t1)
            del L

            print "Computing GMFPT"
            g = GMFPT_theory(D,evals,evecs,weighted=False)
            del evals,evecs,D

            #g_bins = g_bins[np.where(g<=thresh)]
            #g = g[np.where(g<=thresh)]

            # Nan the non-reachable regions
            thresh = np.percentile(g,95)
            #g[np.where(g<=thresh)] = np.nan
            bins = bins[np.where(g<=thresh)]
            g = g[np.where(g<=thresh)]

            myplot(g,bins)
            plt.title(chrom)
            plt.show()
            '''            
            out = np.c_[[chrom for i in xrange(g.shape[0])],\
                        bins*res,\
                        (bins+1)*res,\
                        g]

            g_out = np.array(["%.4f"%(np.nan) for i in xrange(n)],dtype = "S21")

            for bin,value in zip(bins,g):
                g_out[bin] = "%d"%(value)
            del g    
            out = np.c_[[chrom for i in xrange(n)],\
                  np.arange(n)*res,\
                  np.arange(1,n+1)*res,\
                        g_out]
            '''
            #out = out.astype(np.str)

            #np.savetxt(f,out,fmt = "%s",delimiter = "\t")
            write_feature_file(f,data = (chrom,bins,g),res=res,feature_fmt = "%.f")
            print "%s finished"%chrom

        print "Everything is finished" 
        


def GMFPT_theory(d,l,v,weighted=True,select = "all") :
    """
    According to the theory of Lin et al., 2012, the global mean first passage
    time can be calculated by finding the eigenspectrum of the Laplacian matrix
    of the graph. This function calculates the GMFPT from their formula, for the
    graph described by the adjacency matrix A, to all sites. Optional parameter
    'weighted' allows for the choice of having the same quantity but weighted
    with the stationary distribution.
    
    arguments:
        d - degree vector
        l - eigenvalues
        v - eigenvectors
        select - which eigenvectors to select
    """
    
    N = d.shape[0]
    E = np.sum(d)/2.
    
    sortidx = np.argsort(l)
    if select != 'all':
        sortidx = sortidx[select]
        print sortidx.shape
    l = l[sortidx]
    v = v[:,sortidx]
    T = np.zeros(N)
    dv = np.dot (v.T,d)

    if not weighted :
        T = np.dot(2.0*E*v[:,1:]**2 - v[:,1:]*dv[1:],l[1:]**-1)
        return float(N)/(N-1.0) * T
    else :
        T = np.dot((2.0*E)**2*v[:,1:]**2 - 2*(2*E)*v[:,1:]**2*dv[1:]\
                   - (v[:,1:]*dv[1:])**2,l[1:]**-1)
        return T/(2.*E)

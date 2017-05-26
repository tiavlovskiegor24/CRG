import numpy as np
import matplotlib.pyplot as plt
from myplot import myplot
import cooler
from scipy import sparse
from feature_file_processing import write_feature_file
from hi_c_cooler import hi_c_cooler

def Contact_Decay_Feature(newfilename,filepath,res = None,chrom = None):
    # choose resolution file and chromosome
    #filepath = "./matrices/Jurkat/C025_C029II_Jurkat_WT_Q20_50kb.cool"
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
            print type(chrom)

            #cis = sparse.csr_matrix(c.matrix(balance = False)\
             #                       .fetch(chrom)).astype(np.float)

            cis,bins = hi_c_cooler(filepath,chrom,res = res)
            n = cis.shape[0]
            try:
                dec_vec = gen_dec_feature_vec(cis)
            except:
                continue

            myplot(dec_vec)
            plt.title(chrom)
            plt.show()

            '''
            dec_vec = ["%.4f"%(i) for i in dec_vec]

            out = np.c_[np.array([chrom for i in xrange(n)],dtype = np.str),\
                  np.arange(n)*res,\
                  np.arange(1,n+1)*res,\
                  dec_vec]
            out = out.astype(np.str)
            '''
            #np.savetxt(f,out,fmt = "%s",delimiter = "\t")

            write_feature_file(f,data =(chrom,bins,dec_vec),res = res,feature_fmt = "%.3f")
            print "%s finished\n"%chrom

        print "Everything is finished" 

def process_data(data):
    data = (data+1)*1./(data+1).sum()# probability distribution using\
               #Laplace rule
    #data = np.cumsum(data[::-1])[::-1]#cum pd from the tail
    return data

def gen_dec_feature_vec(M,dbins_max = 100):
    '''
    
    '''
    
    n = M.shape[0]
    dec_vec = np.zeros(n)
    dbins = np.arange(1,dbins_max+1)

    # first row
    data = M[0].toarray().ravel()[1:dbins_max+1]
    data = process_data(data)
    dec_vec[0] = compute_decay(data,dbins,plot = False)[0]
    #plt.title("bin %d, decay rate = %.2f"%(0,dec_vec[0]))

    # last row
    data = M[n-1].toarray().ravel()[::-1][1:dbins_max+1]
    data = process_data(data)   
    dec_vec[n-1] = compute_decay(data,dbins,plot = False)[0]
    #plt.title("bin %d, decay rate = %.2f"%(n-1,dec_vec[n-1]))

    if n % 2 != 0:
        #middle row of odd number of rows
        data = M[n/2+1].toarray().ravel()[1:dbins_max+1]
        data = process_data(data)   
        dec_vec[n/2-1] = compute_decay(data,dbins,plot = False)[0]
    

    #sample = np.sort(np.random.randint(1,n,1)).tolist()[::-1]
    sample = []

    for loc in xrange(1,n/2):
        for select in [loc,-loc-1]:

            data = M[select].toarray().ravel()
            if select == -loc-1:
                data = data[::-1]

            data[loc+1:2*loc+1] = np.maximum(data[:loc][::-1],\
                                             data[loc+1:2*loc+1])
            data = data[loc+1:]
            data = process_data(data)
            data = data[:dbins_max]

            if select in sample or n+select in sample:
                 dec_vec[select] = compute_decay(data,dbins,plot=False)[0]
                 #plt.title("bin %d, decay rate = %.2f"%(select,dec_vec[select]))
                 sample.pop()
            else:
                dec_vec[select] = compute_decay(data,dbins)[0]
         
    return dec_vec
    

def compute_decay(contacts,dbins = None,plot=False):
    '''
    function takes raw hic matrix row and computes a power law decay rate of contacts with respect to separation in bins
    '''

    # copute contact probabilities
    #p = contacts*1./contacts.sum()
    #print p
    
    if dbins is None:
        dbins = np.arange(1,contacts.shape[0]+1)

    
    fit = np.polyfit(np.log10(dbins),np.log10(contacts),1)
    fit_fn = np.poly1d(fit) 

    if plot:
        f,ax = myplot(np.log10(contacts),np.log10(dbins),style = 'yo')
        ax.plot(np.log10(dbins), fit_fn(np.log10(dbins)), '--k')

    return fit
    


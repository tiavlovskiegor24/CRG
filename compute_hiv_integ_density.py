import numpy as np
import pandas as pd
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


def compute_hiv_integ_density(newfilename,source,res = None, chrom = None):

    if res is None:
        print "Indicate resolution"
        return

    df = pd.read_table(source,comment = "#",low_memory = False) # load the bhive dataset
    df = df[["insert_position","chr"]] # leave only integration site position and chromosome columns
    df["bin"] = (df["insert_position"].values / res).astype(int) # discretise the position to bin resolution

    # construct dictionary of chromosome indices
    chrom_sort = {} 
    for chrom in df["chr"].unique():
        chrom_sort[chrom] = np.where(df["chr"] == chrom)[0]

    with open(newfilename,"wb") as f: 
        print "Processing {}".format(chrom)

        for chrom in chrom_sort:
            integ_counts = df.iloc[chrom_sort[chrom],:].groupby("bin").agg("count")
            bins = integ_counts.index.values
            integ_counts= integ_counts["insert_position"].values


            fig,ax = myplot()
            ax.vlines(bins,0,integ_counts)
            plt.title(chrom)
            plt.show()

            if chrom[-1] == "_" or chrom in {"chrM","chrUn"}:
                #chrom = chrom[:-1]
                continue

            write_feature_file(f,data = (chrom,bins,integ_counts),res = res,feature_fmt = "%d")
            print "{} finished\n".format(chrom)

    print "Everything is finished" 

import numpy as np
import pandas as pd
from feature_file_processing import read_feature_file, full_array,ravel_feature_data
from myplot import myplot
import sys
from dataset_processing import import_features,train_test_split

def create_full_hiv_expr_dataset(newfilename,seed_dataset = None,res = "50kb",train_test = True):

    resolution = {'100kb': 100000, '10kb': 10000, '500kb': 500000, '50kb': 50000,"":None,'5kb':5000}

    if seed_dataset is None:
        print "Please, provide the seed dataset"
        return
    seed_dataset = seed_dataset.format(res)
    print "Loading seed dataset: {}\n".format(seed_dataset)
    df = pd.read_table(seed_dataset)
    #df["bin"] = df["pos"].values.astype(int)/resolution[res]
    
    #import additional features from files
    print "\nImporting features from files"
    resolution = "50kb"# select from "10kb","50kb","100kb" and "500kb"
    directory = "/mnt/shared/data/HiC_processing/"
    feature_filenames = {"contact_decay":"contacts_decay_Jurkat_",
                         #"gmfpt":"gmfpt_feature_Jurkat_",
                         "row_sum":"row_sum_Jurkat_",
                         "ab_score":"ab_score_Jurkat_",
                         "gc_content":"gc_content_Jurkat_",
                         "lamin":"lamin_Jurkat_",                         
    }
    
    df = import_features(df,"bin",res,directory,feature_filenames)    

    # get target values from the dataset
    #print "\nForming target values"
    #y = get_target_values(df)
    #df = drop_features(df,["RNA","DNA"])
    #df["targets"] = y
    
    # get chromosome indices
    chrom_sort = {}
    for chrom in df["chrom"].unique():
        chrom_sort[chrom] = np.where(df["chrom"] == chrom)[0]

    # adding control_targets column
    df["control_targets"] = np.sum(df[["row_sum","contact_decay"]].values,axis = 1)


    # getting chip_z25 features
    print "Importing ChIP_z25 features from {}".format("data/ChIP_features_{}.txt".format(res))
    df_other = pd.read_table("data/ChIP_features_{}.txt".format(res))
    #df_other["bin"] = (df_other["start"].values / resolution[res]).astype(int) # discretise the position to bin resolution
    columns_to_drop = ["start","end","chrom","pos"]
    
    skip_features = df.columns.tolist()
    present_features = set(skip_features+columns_to_drop)

    chroms_bins = map(tuple,df_other[["chrom","bin"]].values.tolist())
    '''
    chroms = {}
    for chrom in np.unique(df_other["chrom"]):
        indices = np.where(df_other["chrom"].values == chrom)[0]
        bins = df_other["bin"].values[indices]
        #for b in np.unique(df_other["bin"].values[indices]):
         #   bins[b] = np.where(df_other["bin"].values[indices] == b)[0]
        chroms[chrom] = (indices,bins)
    '''
    target_indices = {}
    for i,(chrom,b) in enumerate(df[["chrom","bin"]].values):
        if (chrom,b) in target_indices:
            target_indices[(chrom,b)].append(i)
        else:
            target_indices[(chrom,b)] = [i]
        
    sys.stderr = open("stderr.txt","w")
    
    for column in df_other.columns.tolist():
        if column not in present_features and column[:2] != "d_":
            print column
            
            feature_vector = np.ones(df.shape[0])*np.nan

            values = df_other[column].values
            for chrom_bin,value in zip(chroms_bins,values):
                    
                if chrom_bin not in target_indices:
                    #t_idx = np.where((df["chrom"].values == chrom) & (df["bin"] == b))[0]
                    #target_indices[(chrom,b)] = t_idx\
                    continue
                else:
                    t_idx = target_indices[chrom_bin]
                    feature_vector[t_idx] = value                    
                 
            df[column] = feature_vector
            if column == "H3K27me3_GSE59257_z25":
                for tup in df[["chrom","bin",column]].values:
                    #if tup[-1] is np.nan:
                    print >>sys.stderr, "{}".format(tup)
                        
            print np.isnan(df[column].values).sum()
            print

    del df_other,feature_vector,target_indices

    # getting chip_c features
    print "Importing ChIP-C features from {}".format("data/Jurkat_chip_c_{}.txt".format(res))
    df_other = pd.read_table("data/Jurkat_chip_c_{}.txt".format(res))
    #df_other["bin"] = (df_other["start"].values / resolution[res]).astype(int) # discretise the position to bin resolution
    columns_to_drop = ["start","end","chrom","pos"]
    
    skip_features = df.columns.tolist()
    present_features = set(skip_features+columns_to_drop)

    chroms_bins = map(tuple,df_other[["chrom","bin"]].values.tolist())
    '''
    chroms = {}
    for chrom in np.unique(df_other["chrom"]):
        indices = np.where(df_other["chrom"].values == chrom)[0]
        bins = df_other["bin"].values[indices]
        #for b in np.unique(df_other["bin"].values[indices]):
         #   bins[b] = np.where(df_other["bin"].values[indices] == b)[0]
        chroms[chrom] = (indices,bins)
    '''
    target_indices = {}
    for i,(chrom,b) in enumerate(df[["chrom","bin"]].values):
        if (chrom,b) in target_indices:
            target_indices[(chrom,b)].append(i)
        else:
            target_indices[(chrom,b)] = [i]
    
    for column in df_other.columns.tolist():
        if column not in present_features and column[:2] != "d_":
            print column
            
            feature_vector = np.ones(df.shape[0])*np.nan

            values = df_other[column].values
            for chrom_bin,value in zip(chroms_bins,values):
                    
                if chrom_bin not in target_indices:
                    #t_idx = np.where((df["chrom"].values == chrom) & (df["bin"] == b))[0]
                    #target_indices[(chrom,b)] = t_idx\
                    continue
                else:
                    t_idx = target_indices[chrom_bin]
                    feature_vector[t_idx] = value                    
                 
            df[column] = feature_vector
                        
            print np.isnan(df[column].values).sum()
            print

    del df_other,feature_vector,target_indices

    
    if train_test:
        #split in train and test uniformly from each chromosome
        print "\nSpliting into train and test and saving the datasets"
        
        train_idx,test_idx = train_test_split(df,0.9,chrom_sort)
        for name,idx in zip(["train","test"],[train_idx,test_idx]):

            to_write = df.iloc[idx,:].copy()

            #if name is "test":
             #   to_write = drop_features(to_write,["targets"])

            to_write.to_csv("data/{}_{}_{}.txt".format(newfilename,name,res),sep="\t",index=False,na_rep="nan")
            print "\tdata/{}_{}_{}.txt".format(newfilename,name,res)
    else:
        to_write.to_csv("data/{}_{}_{}.txt".format(newfilename,"full",res),sep="\t",index=False,na_rep="nan")        

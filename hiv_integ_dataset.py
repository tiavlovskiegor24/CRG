import numpy as np
import pandas as pd
from feature_file_processing import read_feature_file, full_array,ravel_feature_data
from myplot import myplot
import sys
from dataset_processing import import_features,train_test_split


def create_full_hiv_integ_dataset(newfilename,seed_dataset = None,res = "50kb",train_test = True):
    '''
    function to create merged dataset staring from seed dataset and adding feature values listed 
    in 'feature_filenames' and stored in 'directory' 
    '''
    resolution = {'100kb': 100000, '10kb': 10000, '500kb': 500000, '50kb': 50000,"":None,'5kb':5000}
    directory = "/mnt/shared/data/HiC_processing/"
    '''
    data = read_feature_file(directory+"hiv_integ_density_bushman_"+res,as_dict = True)
    data = full_array(data,res = resolution[res],fill_value = 0)
    data = ravel_feature_data(data)
    '''

    if seed_dataset is None:
        print "Please, provide the seed dataset"
        return
    seed_dataset = seed_dataset.format(res)
    print "Loading seed dataset: {}\n".format(seed_dataset)
    df = pd.read_table(seed_dataset)
    #df["bin"] = (df["start"].values / resolution[res]).astype(int) # discretise the position to bin resolution

    '''
    df = pd.DataFrame()
    df["bin"] = (data["bin_start"]/resolution[res]).astype(int)
    df["integ_density"] = data["value"]
    df["chrom"] = data["chrom"]
    del data
    '''

    #import additional features from files
    print "\nImporting features from files"
    #resolution = "5kb"# select from "10kb","50kb","100kb" and "500kb"
    feature_filenames = {"contact_decay":"contacts_decay_Jurkat_",
                         #"gmfpt":"gmfpt_feature_Jurkat_",
                         "row_sum":"row_sum_Jurkat_",
                         "ab_score":"ab_score_Jurkat_",
                         "integ_density":"hiv_integ_density_bushman_",
                         "gc_content":"gc_content_Jurkat_",
                         "lamin":"lamin_Jurkat_"
    }
    
    
    df = import_features(df,"bin",res,directory,feature_filenames)    
    
    # get chromosome indices
    chrom_sort = {}
    for chrom in df["chrom"].unique():
        chrom_sort[chrom] = np.where(df["chrom"] == chrom)[0]

    # adding control_targets column
    df["control_targets"] = np.sum(df[["row_sum","contact_decay"]].values,axis = 1)

    '''
    # getting all feature of jurkat
    df_jurkat = pd.read_table("data/Jurkat_hiv_50kb.txt",comment = "#")
    df_jurkat["bin"] = (df_jurkat["pos"].values / 50000).astype(int) # discretise the position to bin resolution
    columns_to_drop = ['brcd','pos', 'gene_name',"rep",\
                           "expr","nread","mapq","strand","cat","chrom","DNA","RNA"]
    
    skip_features = df.columns.tolist()
    present_features = set(skip_features+columns_to_drop)
    chroms = {}
    for chrom in np.unique(df_jurkat["chrom"]):
        indices = np.where(df_jurkat["chrom"].values == chrom)[0]
        bins = {}
        for b in np.unique(df_jurkat["bin"].values[indices]):
            bins[b] = np.where(df_jurkat["bin"].values[indices] == b)[0]
        chroms[chrom] = (indices,bins)

    target_indices = {}
    for i,(chrom,b) in enumerate(df[["chrom","bin"]].values):
        target_indices[(chrom,b)] = i
        
    for column in df_jurkat.columns.tolist():
        if column not in present_features and column[:2] != "d_":
            print column
            
            feature_vector = np.ones(df.shape[0])*np.nan
            
            for chrom,(indices,bins) in chroms.iteritems():
                for b,b_indices in bins.iteritems():
                    bin_average = np.mean(df_jurkat[column].values[indices][b_indices])
                    if (chrom,b) not in target_indices:
                        t_idx = np.where((df["chrom"].values == chrom) & (df["bin"] == b))[0]
                        target_indices[(chrom,b)] = t_idx 
                    else:
                        t_idx = target_indices[(chrom,b)]
                    feature_vector[t_idx] = bin_average
                   
                 
            df[column] = feature_vector
            print np.isnan(df[column].values).sum()
            print

    del df_jurkat,feature_vector,target_indices

    # getting all data from random genome sample
    df_jurkat = pd.read_table("data/Jurkat_gws_50kb.txt",comment = "#")
    df_jurkat["bin"] = (df_jurkat["beg"].values / 50000).astype(int) # discretise the position to bin resolution
    columns_to_drop = ['brcd','beg','end','id', 'gene_name',"rep",\
                           "expr","nread","mapq","strand","cat","chr","DNA","RNA"]
    
    
    present_features = set(skip_features+columns_to_drop)
    chroms = {}
    for chrom in np.unique(df_jurkat["chr"]):
        indices = np.where(df_jurkat["chr"].values == chrom)[0]
        bins = {}
        for b in np.unique(df_jurkat["bin"].values[indices]):
            bins[b] = np.where(df_jurkat["bin"].values[indices] == b)[0]
        chroms[chrom] = (indices,bins)

    target_indices = {}
    for i,(chrom,b) in enumerate(df[["chrom","bin"]].values):
        target_indices[(chrom,b)] = i
        
    for column in df_jurkat.columns.tolist():
        if column not in present_features and column[:2] != "d_":
            print column
            
            feature_vector = np.ones(df.shape[0])*np.nan
            
            for chrom,(indices,bins) in chroms.iteritems():
                for b,b_indices in bins.iteritems():
                    bin_average = np.mean(df_jurkat[column].values[indices][b_indices])
                    if (chrom,b) not in target_indices:
                        t_idx = np.where((df["chrom"].values == chrom) & (df["bin"] == b))[0]
                        target_indices[(chrom,b)] = t_idx 
                    else:
                        t_idx = target_indices[(chrom,b)]
                    feature_vector[t_idx] = bin_average
                    
            #df[column] = np.where(np.isnan(feature_vector),df[column.values],feature_vector)
            print (np.isnan(df[column].values) & ~np.isnan(feature_vector)\
                | ~np.isnan(df[column].values) & np.isnan(feature_vector)).sum()
            
            df[column] = np.where(np.isnan(df[column].values),feature_vector,df[column].values)
            print np.isnan(df[column].values).sum()
            print 
            
    del df_jurkat,feature_vector,target_indices
    

    # getting chip_z25 features
    df_jurkat = pd.read_table("data/ChIP_features.txt")
    df_jurkat["bin"] = (df_jurkat["start"].values / resolution[res]).astype(int) # discretise the position to bin resolution
    columns_to_drop = ["start","end","chrom"]
    
    skip_features = df.columns.tolist()
    present_features = set(skip_features+columns_to_drop)
    chroms = {}
    for chrom in np.unique(df_jurkat["chrom"]):
        indices = np.where(df_jurkat["chrom"].values == chrom)[0]
        bins = df_jurkat["bin"].values[indices]
        #for b in np.unique(df_jurkat["bin"].values[indices]):
         #   bins[b] = np.where(df_jurkat["bin"].values[indices] == b)[0]
        chroms[chrom] = (indices,bins)

    target_indices = {}
    for i,(chrom,b) in enumerate(df[["chrom","bin"]].values):
        target_indices[(chrom,b)] = i
        
    for column in df_jurkat.columns.tolist():
        if column not in present_features and column[:2] != "d_":
            print column
            
            feature_vector = np.ones(df.shape[0])*np.nan
            
            for chrom,(indices,bins) in chroms.iteritems():
                values = df_jurkat[column].values[indices]

                for value,b in zip(values,bins):
                    
                    if (chrom,b) not in target_indices:
                        t_idx = np.where((df["chrom"].values == chrom) & (df["bin"] == b))[0]
                        target_indices[(chrom,b)] = t_idx 
                    else:
                        t_idx = target_indices[(chrom,b)]
                        feature_vector[t_idx] = value
                   
                 
            df[column] = feature_vector
            print np.isnan(df[column].values).sum()
            print

    del df_jurkat,feature_vector,target_indices
    '''
    
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


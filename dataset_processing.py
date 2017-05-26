import numpy as np
import pandas as pd
from feature_file_processing import read_feature_file, full_array,ravel_feature_data
from myplot import myplot


def create_full_hiv_expr_dataset(filename,train_test = True):

    #read the dataframe from file
    df = pd.read_table(filename,comment = "#")

    #import additional features from files
    print "\nImporting features from files"
    resolution = "50kb"# select from "10kb","50kb","100kb" and "500kb"
    directory = "/mnt/shared/data/HiC_processing/"
    feature_filenames = {"contact_decay":"contacts_decay_Jurkat_",
                         "gmfpt":"gmfpt_feature_Jurkat_",
                         "row_sum":"row_sum_Jurkat_",
    }
    
    df = import_features(df,resolution,directory,feature_filenames)    

    # get target values from the dataset
    #print "\nForming target values"
    #y = get_target_values(df)
    #df = drop_features(df,["RNA","DNA"])
    #df["targets"] = y
    
    # get chromosome indices
    chrom_sort = {}
    for chrom in df["chrom"].unique():
        chrom_sort[chrom] = np.where(df["chrom"] == chrom)[0]

    #df = drop_features(df,["chrom"])

    if train_test:
        #split in train and test uniformly from each chromosome
        print "\nSpliting into train and test and saving the datasets"
        
        train_idx,test_idx = train_test_split(df,0.9,chrom_sort)
        for name,idx in zip(["train","test"],[train_idx,test_idx]):

            to_write = df.iloc[idx,:].copy()

            #if name is "test":
             #   to_write = drop_features(to_write,["targets"])

            to_write.to_csv("data/Jurkat_hiv_{}_50kb.txt".format(name),sep="\t",index=False)
    else:
        to_write.to_csv("data/Jurkat_hiv_{}_50kb.txt".format("full"),sep="\t",index=False)        


def create_full_hiv_integ_dataset(newfilename,res = "5kb",train_test = True):

    resolution = {'100kb': 100000, '10kb': 10000, '500kb': 500000, '50kb': 50000,"":None,'5kb':5000}
    directory = "/mnt/shared/data/HiC_processing/"

    data = read_feature_file(directory+"hiv_integ_density_large_"+res,as_dict = True)
    data = full_array(data,res = resolution[res],fill_value = 0)
    data = ravel_feature_data(data)
    
    
    df = pd.DataFrame()
    df["bin"] = (data["bin_start"]/resolution[res]).astype(int)
    df["integ_density"] = data["value"]
    df["chrom"] = data["chrom"]
    del data
    
    #import additional features from files
    print "\nImporting features from files"
    #resolution = "5kb"# select from "10kb","50kb","100kb" and "500kb"
    feature_filenames = {"contact_decay":"contacts_decay_Jurkat_",
                         #"gmfpt":"gmfpt_feature_Jurkat_",
                         "row_sum":"row_sum_Jurkat_",
                         #"ab_score":"ab_score_",
                         "integ_density":"hiv_integ_density_large_",
    }
    
    
    df = import_features(df,"bin",res,directory,feature_filenames)    
    print df.shape
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
    
    '''

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

    
    if train_test:
        #split in train and test uniformly from each chromosome
        print "\nSpliting into train and test and saving the datasets"
        
        train_idx,test_idx = train_test_split(df,0.9,chrom_sort)
        for name,idx in zip(["train","test"],[train_idx,test_idx]):

            to_write = df.iloc[idx,:].copy()

            #if name is "test":
             #   to_write = drop_features(to_write,["targets"])

            to_write.to_csv("data/{}_{}_50kb.txt".format(newfilename,name),sep="\t",index=False)
    else:
        to_write.to_csv("data/{}_{}_50kb.txt".format(newfilename,"full"),sep="\t",index=False)        

        
def train_test_split(df,train_f = 0.9,chrom_sort = None):
    '''
    by default chrom_sort is passed and the dataset is split
    into train and test datasets with balanced number of samples for each chromosome
    ''' 
    if chrom_sort is not None:
        train_idx = np.array([],dtype=np.int) 
        test_idx = np.array([],dtype=np.int)

        for chrom,idx in chrom_sort.iteritems():
            n = idx.shape[0]
            s_idx = np.zeros_like(idx,dtype = bool)
            s_idx[np.random.choice(n,int(n*train_f),replace=False)] = True
            train_idx= np.r_[train_idx,idx[s_idx]]
            test_idx = np.r_[test_idx,idx[~s_idx]]
    else:
        n = df.shape[0]
        s_idx = np.zeros(n,dtype = bool)
        s_idx[np.random.choice(n,int(n*train_f),replace=False)] = True
        train_idx = np.arange(n)[s_indx]
        test_idx = np.arange(n)[~s_idx]
        
    return train_idx,test_idx



def import_features(df,map_col = "pos",res = "",directory = None,feature_filenames = None):
    resolution = {'100kb': 100000, '10kb': 10000, '500kb': 500000, '50kb': 50000,"":None,"5kb":5000}

    if map_col == "bin":
        bins = df[map_col].values.astype(np.int)
    elif map_col == "pos":
        bins = (df[map_col]/resolution[res]).astype(np.int).values
    else:
        print "INvalid mapping column name"
        return

    for feature in feature_filenames:
        #print "Computing feature: %s"%feature
        print "\tImporting feature: {}".format(feature)
        #data = np.loadtxt(directory+feature_filenames[feature]+res,dtype=np.str,delimiter="\t")
        data = read_feature_file(directory+feature_filenames[feature]+res)
        data = full_array(data,res = resolution[res])
 
        df[feature] = df[map_col]*0
        if isinstance(data,dict):
            for chrom in df["chrom"].unique():
                idx = np.where(df["chrom"] == chrom)[0]
                if chrom not in data:
                    print "Alarm ",feature,chrom
                    continue
                try:
                    df.ix[idx,feature] = data[chrom]["value"][bins[idx]]
                except:
                    print chrom,bins,idx,data[chrom]["value"]
        else:
            for chrom in df["chrom"].unique():
                idx = np.where(df["chrom"] == chrom)[0]
                if chrom not in np.unique(data[:,0]):
                    print "Alarm ",feature
                df.ix[idx,feature] = data[np.where(data[:,0] == chrom)[0],3][bins[idx]].astype(np.float)
    return df



def drop_features(df_modified,columns_to_drop=None):
    if columns_to_drop is None:
        columns_to_drop = ['brcd','pos', 'gene_name',"rep",\
                           "expr","nread","mapq"]
    
    
    df_modified.drop(columns_to_drop, inplace=True, axis = 1)

    return df_modified

    

def load_feature_names(filename):
    features = {}
    with open(filename,"r") as f:
        for line in f:
            line = line.strip("\n").split("\t")
            feature = [word for word in line[0].split(" ") \
                       if word != ""][1]
            features[feature] = line[1]
    return features


def encode_one_hot(df,cat_features = None):
    if cat_features is None:
        return df
    
    for feature in cat_features:
        f_values = np.unique(df[feature]) 
        print '\n\tThere are {} unique values for "{}" feature.'.format(f_values.shape,feature)
        print "\t",f_values 
        dummies = pd.get_dummies(df[feature],prefix = feature+"_oh_",drop_first = False)

        df = pd.concat([df,dummies],axis = 1)

    return df





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
    feature_filenames = {"contact_decay":"contacts_decay_Jurkat_",\
                         "gmfpt":"gmfpt_feature_Jurkat_",\
                         "row_sum":"row_sum_Jurkat_"}
    
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


def create_full_hiv_integ_dataset(newfilename,res = "50kb",train_test = True):

    resolution = {'100kb': 100000, '10kb': 10000, '500kb': 50000, '50kb': 50000,"":None}
    directory = "/mnt/shared/data/HiC_processing/"

    data = read_feature_file(directory+"hiv_integ_density_Jurkat_"+res,as_dict = True)
    data = full_array(data,res = resolution[res],fill_value = 0)
    data = ravel_feature_data(data)
    print data
    
    df = pd.DataFrame()
    df["bins"] = (data["bin_start"]/resolution[res]).astype(int)
    df["integ_density"] = data["value"]
    df["chrom"] = data["chrom"]
    print df.head()
    del data
    
    #import additional features from files
    print "\nImporting features from files"
    resolution = "50kb"# select from "10kb","50kb","100kb" and "500kb"
    feature_filenames = {"contact_decay":"contacts_decay_Jurkat_",
                         "gmfpt":"gmfpt_feature_Jurkat_",
                         "row_sum":"row_sum_Jurkat_",
                         #"integ_density":"hiv_integ_density_Jurkat_",
    }
    
    
    df = import_features(df,"bins",res,directory,feature_filenames)    

    # get chromosome indices
    chrom_sort = {}
    for chrom in df["chrom"].unique():
        chrom_sort[chrom] = np.where(df["chrom"] == chrom)[0]


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
    resolution = {'100kb': 100000, '10kb': 10000, '500kb': 50000, '50kb': 50000,"":None}

    if map_col == "bins":
        bins = df[map_col].values.astype(np.int)
    else:
        bins = (df[map_col]/resolution[res]).astype(np.int).values

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
                    print "Alarm ",feature
                df.ix[idx,feature] = data[chrom]["value"][bins[idx]]                
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





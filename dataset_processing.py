import numpy as np
import pandas as pd
from feature_file_processing import read_feature_file, full_array
from myplot import myplot
from collections import namedtuple


class ML_inputs_tuple(object):
    
    def __init__(self,ML_inputs_dict):
        self.data = namedtuple("ML_inputs",ML_inputs_dict.keys())(**ML_inputs_dict)
    
    def __getitem__(self,key):
        return getattr(self.data,key)

    def __repr__(self):
        return self.data.__repr__()


def get_ML_inputs(source,sample_groups = None,cat = "",feature_types = None):

    #create dictionary of relevant inputs to Machine Learning pipeline
    ML_inputs = dict(filename = None,category=cat,features = None,targets = None,\
                     sample_groups = None,feature_names = None,feature_types = None)
    
    
    if isinstance(source,str):
        #read the dataframe from file
        df = pd.read_table(source,comment = "#")
        ML_inputs["filename"] = source
        
    elif isinstance(source,pd.DataFrame):
        df = source
    else:
        print "Invalid source input\n Must be pandas dataset or csv filename"
        return None

    while cat not in ["train","test"]:
        cat = raw_input("Indicate 'train' or 'test': ")

    m,n = df.shape # m is number of samples and n is number of features
    
    # form index arrays for each of the sample groups
    groups = ["chrom"]
    if sample_groups is not None:
        ML_inputs["groups"] = {}
        for feature in sample_groups:
            # get group indices
            if feature in df.columns.tolist():
                group_sort = {}
                for group_name in df[feature].unique():
                    group_sort[group_name] = np.where(df[feature] == group_name)[0]

                df = drop_features(df,[feature])

                ML_inputs["sample_groups"][feature] = group_sort
            else:
                continue

                    
    #enconde categorical features using one hot encoding
    print "\nEncoding categorical values"
    cat_features = ["cat","strand"]
    df = encode_one_hot(df,cat_features)
    df = drop_features(df,cat_features)

    # get target values from the dataset 
    print "\nExtracting target values"
    targets = df["targets"]
    df = drop_features(df,["targets"])

    ML_inputs["targets"] = targets

    feature_names = df.columns.tolist()
    ML_inputs["feature_names"] = feature_names
    
    # from index arrays for each feature type
    f_types = {"distances":{"id_fun":(lambda x:True if x[:2]=="d_" else False),"preprocess":np.log1p}}
    
    if f_types is not None:
        ML_inputs["feature_types"] = {}
       

        for f_type in f_types:
            id_fun = f_types[f_type]["id_fun"]
            ML_inputs["feature_types"][f_type] = [i for i in xrange(n) if id_fun(feature_names[i])]
            
    #Converting dataset to numpy matrix 
    print "\nExctacting feature matrix"
    features = df.as_matrix().astype(np.float)


    # preprocess features of specific types
    if f_types is not None:
        print "\nPreprocessing feature values"
        for f_type in f_types:
            idx = ML_inputs["feature_types"][f_type]
            p_fun = f_types[f_type]["preprocess"]
            print "\tApplying '{}' to '{}' features".format(p_fun.__name__,f_type)
            df.iloc[:,idx] = p_fun(df.iloc[:,idx].values)
            

    ML_inputs["features"] = features

    #convert inputs to tuple
    ML_inputs = ML_inputs_tuple(ML_inputs)
    
    return ML_inputs


def create_full_dataset(filename,train_test = True):

    #read the dataframe from file
    df = pd.read_table(filename,comment = "#")

    #import additional features from files
    print "\nImporting features from files"
    resolution = "50kb"# select from "10kb","50kb","100kb" and "500kb"
    directory = "/mnt/shared/data/HiC_processing/"
    feature_filenames = {"c_decay":"contacts_decay_Jurkat_",\
                         "gmfpt":"gmfpt_feature_Jurkat_",\
                         "row_sum":"row_sum_Jurkat_"}
    
    df = import_features(df,resolution,directory,feature_filenames)    

    #drop any other features
    print "\nDropping features"
    columns_to_drop = ['brcd','pos', 'gene_name',"rep",\
                           "expr","nread","mapq"]

    df = drop_features(df,columns_to_drop)

    #handling Nans
    print "\nHandling NaNs"
    nan_samples = (df.isnull().values.sum(axis = 1) > 0)
    print "\tDropping {} samples with NaN entries".format(nan_samples.sum())
    df = df.ix[~nan_samples,:]
    print df.isnull().values.sum()

    # get target values from the dataset
    print "\nForming target values"
    y = get_target_values(df)
    df = drop_features(df,["RNA","DNA"])
    df["targets"] = y
    
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

            to_write.to_csv("Jurkat_hiv_{}_50kb.txt".format(name),sep="\t",index=False)
    else:
        to_write.to_csv("Jurkat_hiv_{}_50kb.txt".format("full"),sep="\t",index=False)        
                
            
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


def import_features(df,res = "",directory = None,feature_filenames = None):
    resolution = {'100kb': 100000, '10kb': 10000, '500kb': 50000, '50kb': 50000,"":None}
    bins = (df["pos"]/resolution[res]).astype(np.int).values

    for feature in feature_filenames:
        #print "Computing feature: %s"%feature
        print "\tImporting feature: {}".format(feature)
        #data = np.loadtxt(directory+feature_filenames[feature]+res,dtype=np.str,delimiter="\t")
        data = read_feature_file(directory+feature_filenames[feature]+res)
        data = full_array(data,res = resolution[res])
        
        df[feature] = df["pos"]*0
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

    
def get_target_values(df_modified,binary = True):
    # currently target values are asumed
    exp_ratio = df_modified["RNA"]*1.0/df_modified["DNA"]
    
    if binary:
        threshold = 3
        target = np.where(exp_ratio.values > threshold,1.,0.)
        
        y = target.reshape(-1,).astype(np.float)
        #y = np.array(df_modified["pos_expr"].tolist()).reshape(-1,)\
                         #                         .astype(np.float)
        print "Binary label split: %.2f"%(y.sum()/y.shape)[0]
        #print y.shape

    else:
        y = exp_ratio
        
    #df_modified.drop(["RNA","DNA"], inplace=True, axis = 1)
    
    return y


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
        dummies = pd.get_dummies(df[feature],prefix = feature,drop_first = True)

        df = pd.concat([df,dummies],axis = 1)
        #df.drop(feature,inplace = True,axis = 1)

    '''
    # category of the intergration site
    feature = "cat"
    print 'There are ' + str(np.unique(df[feature]).shape[0]) \
        +' unique values for "%s" feature.\n'%feature
    print np.unique(df[feature])
    dummies = pd.get_dummies(df[feature])

    #print "Unique",dummies.sum(axis = 1).unique()
    dummies.drop("IN",inplace = True,axis = 1)
    df_modified = pd.concat([df,dummies],axis = 1)

    # strand of the integrations site
    feature = "strand"
    print 'There are ' + str(np.unique(df[feature]).shape[0]) \
        +' unique values for "%s" feature.\n'%feature
    print np.unique(df[feature])
    dummies = pd.get_dummies(df[feature])

    #print "Unique",dummies.sum(axis = 1).unique()
    dummies.drop("-",inplace = True,axis = 1)
    dummies.columns = ["Stand +"]
    df_modified = pd.concat([df_modified,dummies],axis = 1)

    # integration site in gene
    df_modified["in_gene"] = np.where(pd.isnull(df_modified["gene_name"])\
                                      ,0.,1.)
    '''
    return df





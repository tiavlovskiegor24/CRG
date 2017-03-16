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

    def get_data(self,mask = True):
        if not mask:
            confirm = raw_input("Are you sure you want to get non-masked data?(Y/n)\n Some samples may contain Nans and some columns may not be suitable for Machine Learning")
            if confirm == "Y":
                samples = getattr(self.data,"samples")
                targets = getattr(self.data,"targets")
                return samples,targets
            
        print "Returning masked data"
        samples_mask,features_mask = getattr(self.data,"mask")
        samples = getattr(self.data,"samples")[samples_mask,:][:,features_mask].astype(np.float)
        targets = getattr(self.data,"targets")[samples_mask].astype(np.float)

        return samples,targets
        
    def __repr__(self):
        return self.data.__repr__()


    
def get_ML_inputs(dataset = None,cat = "",):
    # importing pipeline controls
    from control_file import ml_method,feature_types_to_exclude_list, features_to_exclude_list, sample_groups, source, get_targets

    # importing feature types dictionary
    from feature_types import feature_types_dict as f_types    

    
    #create datastructure of relevant inputs to Machine Learning pipeline
    ML_inputs = dict(filename = None,category=cat,samples = None,targets = None,\
                     sample_groups = None,feature_names = None,feature_types = None,\
                     mask = None)
    
    if dataset is None:
        #read the dataframe from file
        df = pd.read_table(source.format(cat),comment = "#")
        ML_inputs["filename"] = source.format(cat)    
    elif isinstance(dataset,pd.DataFrame):
        df = source
    else:
        print "Invalid source input\n Must be pandas dataset or csv filename"
        return None

    
    while cat not in ["train","test"]:
        cat = raw_input("Indicate 'train' or 'test': ")

    # Track the steps of the routine
    step_tracker = 0

    #enconde categorical features using one hot encoding
    step_tracker += 1
    print "\n\n{}. Encoding categorical features".format(step_tracker)
    cat_features = ["cat","strand"]
    df = encode_one_hot(df,cat_features)
    df = drop_features(df,cat_features)

    # get the final df shape
    m,n = df.shape # m - number of samples,n - number of features

    # default matrix mask are full range
    features_mask = np.arange(n)
    samples_mask = np.arange(m)

    
    feature_names = df.columns.tolist()
    ML_inputs["feature_names"] = np.array(feature_names,dtype=str)
    
    
    # identify all features_types present in dataset and removing the ones in drop list
    step_tracker += 1
    print "\n\n{}. Identifying present feature types in the dataset".format(step_tracker)
    ML_inputs["feature_types"] = {}
    b_mask = np.ones(n,dtype = bool)
    for f_type in f_types:
        id_fun = f_types[f_type]["id_fun"]
        idx = [i for i in xrange(n) if id_fun(feature_names[i])]
        if idx:
            ML_inputs["feature_types"][f_type] = np.array(idx)
            print "\n\t{} features of type '{}' are present".format(len(idx),f_type)

            if f_type in feature_types_to_exclude_list:
                print "\t\tExcluding feature type: '{}'".format(f_type)
                b_mask[idx] = False

    # update feature mask
    features_mask = features_mask[b_mask]

    del id_fun,idx,b_mask


    #mask individual features to exclude
    step_tracker += 1
    if features_to_exclude_list:
        # exclude targets column by default
        #features_to_exclude_list["targets"] = None
        
        print "\n\n{}. Excluding individual features:\n{}".format(step_tracker,features_to_exclude_list.keys())
        features_mask = [i for i in features_mask if (feature_names[i] not in features_to_exclude_list)]
        

    # preprocess features of specific types
    step_tracker += 1
    print "\n\n{}. Preprocessing feature type values".format(step_tracker)
    for f_type in f_types:
        if f_type in feature_types_to_exclude_list:
            continue
        idx = ML_inputs["feature_types"][f_type]
        p_fun = f_types[f_type]["preprocess"]
        if p_fun is not None:
            print "\n\tPreprocessing '{}' features".format(f_type)
            df.iloc[:,idx] = p_fun(df.iloc[:,idx].values,ml_method)
            nan_samples = np.sum(np.isnan(df.iloc[:,idx].values),axis = 1).sum()
            print "\t{} samples remaining with Nan values for '{}' features".format(nan_samples,f_type)
    del idx,p_fun,nan_samples


    # get target values from the dataset
    step_tracker += 1
    print "\n\n{}. Extracting target values".format(step_tracker)
    targets = get_targets(df)
    nan_targets = np.isnan(targets).ravel()
    print "\t{} targets remaining with Nan values".format(nan_targets.sum(),f_type)
    
    ML_inputs["targets"] = targets

    
                         
    #removing all the remaining Nans
    step_tracker += 1
    nan_samples = (df.iloc[:,features_mask].isnull().values.sum(axis = 1) > 0).ravel()
    nan_samples = np.logical_or(nan_samples,nan_targets)
    print "\n\n{}. Excluding total of {} samples with NaN entries".format(step_tracker,nan_samples.sum())
    samples_mask = np.arange(m)[~nan_samples]
    

    # form index arrays for each of the sample groups
    if sample_groups:
        ML_inputs["sample_groups"] = {}
        for feature in sample_groups:
            # get group indices
            if feature in df.columns.tolist():
                group_sort = {}
                for group_name in df[feature][samples_mask].unique():
                    group_sort[group_name] = np.where(df[feature][samples_mask] == group_name)[0]

                ML_inputs["sample_groups"][feature] = group_sort
            else:
                continue
    

    #Converting dataset to numpy matrix
    step_tracker += 1
    print "\n\n{}. Exctacting feature matrix".format(step_tracker)
    samples = df.as_matrix()

    ML_inputs["samples"] = samples

    ML_inputs["mask"] = (np.array(samples_mask),np.array(features_mask))

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





import numpy as np
import pandas as pd
from feature_file_processing import read_feature_file, full_array
from myplot import myplot
from collections import namedtuple
from auxiliary_items import ML_inputs_tuple

import control_file as cf
import feature_types as f_types

reload(f_types)
reload(cf)

    
def get_ML_inputs(dataset = None):
    # importing pipeline controls
    #from control_file import ml_method,feature_types_to_exclude_list,\
     #   features_to_exclude_list, sample_groups, source, target_type

    # importing feature types dictionary
    #from feature_types import feature_types as f_types
    #from feature_types import target_types

    
    #create datastructure of relevant inputs to Machine Learning pipeline
    ML_inputs = dict(filename = None,
                     train_samples = None,test_samples = None,
                     train_targets = None,test_targets = None,
                     train_sample_groups = None,test_sample_groups = None,
                     feature_names = None,
                     feature_types = None,
                     mask = None,
                     preprocessing = None)
    
    if dataset is None:
        #read the dataframe from file
        df = pd.read_table(cf.source.format("train"),comment = "#")
        df_test = pd.read_table(cf.source.format("test"),comment = "#")
        ML_inputs["filename"] = cf.source.format("train/test")    
    elif isinstance(dataset,pd.DataFrame):
        df = cf.source
        df_test = None
    else:
        print "Invalid source input\n Must be pandas dataset or csv filename"
        return None

    
    #while cat not in ["train","test"]:
     #   cat = raw_input("Indicate 'train' or 'test': ")

    # Track the steps of the routine
    step_tracker = 0

    #enconde categorical features using one hot encoding
    step_tracker += 1
    print "\n\n{}. Encoding categorical features".format(step_tracker)
    cat_features = ["cat","strand"]
    df = encode_one_hot(df,cat_features)
    df = drop_features(df,cat_features)
    df_test = encode_one_hot(df_test,cat_features)
    df_test = drop_features(df_test,cat_features)

    # get the final df shape
    m,n = df.shape # m - number of samples,n - number of features
    m_test = df_test.shape[0]

    # default matrix mask are full range
    train_samples_mask = np.ones(m,dtype = bool)
    test_samples_mask = np.ones(m_test,dtype = bool)
    features_mask = np.ones(n,dtype = bool)

    
    feature_names = df.columns.tolist()
    ML_inputs["feature_names"] = np.array(feature_names,dtype=str)

    
    #go through all feautures and recognise their feature_type if any
    #skip if feature is in features to exlude list
    #other wise store the feature as unrecognised and report to user
    step_tracker += 1
    print "\n\n{}. Identifying features present in the dataset".format(step_tracker)
    f_type_indices = {}
    unrecognised_features = {}
    for i,feature_name in enumerate(feature_names):
        feature_recognised = False
        for f_type in f_types.feature_types:
            
            if f_types.feature_types[f_type]["id_fun"](feature_name):
                feature_recognised = True

                if f_type in f_type_indices:
                    f_type_indices[f_type].append(i)
                else:
                    f_type_indices[f_type] = [i]

        if feature_name in cf.features_to_exclude_list:
            feature_recognised = True
            features_mask[i] = False

        if not feature_recognised:
            unrecognised_features[feature_name] = i

    for f_type in f_type_indices: 
        f_type_indices[f_type] = np.array(f_type_indices[f_type])
        print "\n\t{} features of type '{}' are present".format(len(f_type_indices[f_type]),f_type)
        if f_type in cf.feature_types_to_exclude_list:
            print "\t\tExcluding feature type: '{}'".format(f_type)
            features_mask[f_type_indices[f_type]] = False

    print "\n\tList of excluded individual features:\n\t{}"\
            .format(cf.features_to_exclude_list.keys())

    if unrecognised_features:
        print "\n\tList of unrecognised features:\n\t{}"\
            .format(unrecognised_features.keys())
    else:
        del unrecognised_features

    ML_inputs["feature_types"] = f_type_indices
    del f_type_indices

    
    # preprocess features of specific types
    step_tracker += 1
    print "\n\n{}. Preprocessing feature type values".format(step_tracker)
    preprocess_funcs = {}
    for f_type in ML_inputs["feature_types"]:
        if f_type in cf.feature_types_to_exclude_list:
            continue
        idx = ML_inputs["feature_types"][f_type]
        p_fun = f_types.feature_types[f_type]["preprocess"]
        if p_fun is not None:
            print "\n\tPreprocessing '{}' features".format(f_type)
            p_fun = p_fun(cf.ml_method)
            df.iloc[:,idx] = p_fun.fit_transform(df.iloc[:,idx].values,skip = False)
            df_test.iloc[:,idx] = p_fun.transform(df_test.iloc[:,idx].values)
            nan_samples = np.sum(np.isnan(df.iloc[:,idx].values),axis = 1).sum()
            test_nan_samples = np.sum(np.isnan(df_test.iloc[:,idx].values),axis = 1).sum()
            print "\t{} train and {} test  samples remaining with Nan values for '{}' features."\
                .format(nan_samples,test_nan_samples,f_type)
        preprocess_funcs[f_type] = p_fun
    del idx,p_fun,nan_samples,test_nan_samples


    
    # get target values from the dataset
    step_tracker += 1
    print "\n\n{}. Extracting target values".format(step_tracker)
    print "\n\tUsing '{}' target type".format(cf.target_type["name"])
    targets_p_fun = f_types.target_types[cf.target_type["name"]](features_mask=features_mask,
                                                                 noise = (0,.1),
                                                      fields = {
                                                          #"gmfpt":None,
                                                          #"ab_score":None,
                                                          #"row_sum":None,
                                                      },
                                                      **cf.target_type["params"])
    train_targets = targets_p_fun.fit_transform(df)
    nan_targets = np.isnan(train_targets).ravel()

    test_targets = targets_p_fun.transform(df_test)
    test_nan_targets = np.isnan(test_targets).ravel()
    print "\n\t{} train and {} test targets remaining with Nan values".format(nan_targets.sum(),test_nan_targets.sum(),f_type)
    
    ML_inputs["train_targets"] = train_targets
    ML_inputs["test_targets"] = test_targets

    preprocess_funcs["targets"] = targets_p_fun

    del train_targets,test_targets

    
                         
    #removing all the remaining Nans
    step_tracker += 1
    nan_samples = (df.iloc[:,features_mask].isnull().values.sum(axis = 1) > 0).ravel()
    nan_samples = np.logical_or(nan_samples,nan_targets)
    test_nan_samples = (df_test.iloc[:,features_mask].isnull().values.sum(axis = 1) > 0).ravel()
    test_nan_samples = np.logical_or(test_nan_samples,test_nan_targets)
    
    print "\n\n{}. Excluding total of {} train and {} test samples with NaN entries"\
        .format(step_tracker,nan_samples.sum(),test_nan_samples.sum())
    train_samples_mask = ~nan_samples
    test_samples_mask = ~test_nan_samples
    

    # form index arrays for each of the sample groups
    if cf.sample_groups:
        ML_inputs["train_sample_groups"] = {}
        ML_inputs["test_sample_groups"] = {}
        for feature in cf.sample_groups:
            # get group indices
            if feature in df.columns.tolist():
                group_sort = {}
                test_group_sort = {}
                for group_name in df[feature][train_samples_mask].unique():
                    group_sort[group_name] = np.where(df[feature][train_samples_mask] == group_name)[0]
                for group_name in df_test[feature][test_samples_mask].unique():
                    test_group_sort[group_name] = np.where(df_test[feature][test_samples_mask] == group_name)[0]
                ML_inputs["train_sample_groups"][feature] = group_sort
                ML_inputs["test_sample_groups"][feature] = test_group_sort
            else:
                continue
    del group_sort,test_group_sort
    

    #Converting dataset to numpy matrix
    step_tracker += 1
    print "\n\n{}. Exctacting feature matrix".format(step_tracker)
    train_samples = df.as_matrix()
    ML_inputs["train_samples"] = train_samples

    test_samples = df_test.as_matrix()
    ML_inputs["test_samples"] = test_samples

    ML_inputs["mask"] = namedtuple("ML_inputs_mask","train_samples_mask,test_samples_mask,features_mask")(**{
        
        "train_samples_mask":np.array(train_samples_mask),
        "test_samples_mask": np.array(test_samples_mask),                              
        "features_mask" : np.array(features_mask),
        
        })


    ML_inputs["preprocessing"] = preprocess_funcs

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





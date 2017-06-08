import numpy as np
import pandas as pd
from collections import namedtuple

import dataset_processing as dp
from feature_file_processing import read_feature_file, full_array
from myplot import myplot
import auxiliary_items as aux


class ML_inputs_tuple(object):
    
    def __init__(self,ML_inputs_dict):
        self.data = namedtuple("ML_inputs",ML_inputs_dict.keys())(**ML_inputs_dict)
    
    def __getitem__(self,key):
        if hasattr(self.data,key):
            return getattr(self.data,key)
        else:
            return None

    def get_data(self,train_or_test = "train",masked = True):
        
        if masked:
            #confirm = raw_input("Are you sure you want to get non-masked data?(Y/n)\n Some samples may contain Nans and some columns may not be suitable for Machine Learning")
            #if confirm == "Y":
            samples = getattr(self.data,"{}_samples".format(train_or_test))
            targets = getattr(self.data,"{}_targets".format(train_or_test))
            return samples,targets
            
        #print "\tReturning masked '{}' data".format(train_or_test)
        masks = getattr(self.data,"mask")
        samples_mask = getattr(masks,"{}_samples_mask".format(train_or_test))
        features_mask = getattr(masks,"features_mask")
        
        samples = getattr(self.data,"{}_samples".format(train_or_test))\
                  [samples_mask,:][:,features_mask].astype(np.float)

        targets = getattr(self.data,"{}_targets".format(train_or_test))\
                  [samples_mask].astype(np.float)

        return samples,targets
        
    def __repr__(self):
        return self.data.__repr__()



def get_ML_inputs(cf = None,f_types = None,t_types = None,dataset = None,verbose = True):
    # importing pipeline controls
    if cf is None:
        import control_file as cf

    if f_types is None:
        import feature_types as f_types

    if t_types is None:
        import target_types as t_types

    #from control_file import ml_method,feature_types_to_exclude_list,\
     #   features_to_exclude_list, sample_groups, source, target_type

    # importing feature types dictionary
    #from feature_types import feature_types as f_types
    #from feature_types import target_types


    
    
    #create datastructure of relevant inputs to Machine Learning pipeline
    ML_inputs = dict(filename = None,
                     train_samples = None,test_samples = None,
                     train_targets = None,test_targets = None,
                     train_sample_weight = None,test_sample_weight = None,
                     train_sample_groups = None,test_sample_groups = None,
                     feature_names = None,
                     feature_types = None,
                     mask = None,
                     preprocessing = None,
    )
    
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

    # select specific group of samples from the full dataset
    if hasattr(cf,"select_sample_group"):
        if cf.select_sample_group:
            step_tracker += 1
            if verbose:
                print "\n\n{}. Selecting a sample group".format(step_tracker)

            for col,condition in cf.select_sample_group.iteritems():
                if verbose:
                    print "\n\t '{}'{}".format(col,condition.strip("x"))
                masking_fun = aux.create_masking_fun(condition)
                df = df[masking_fun(df[col].values)]
                df_test = df_test[masking_fun(df_test[col].values)]


    #enconde categorical features using one hot encoding
    if hasattr(cf,"one_hot_features"):
        step_tracker += 1
        if verbose:
            print "\n\n{}. Encoding categorical features".format(step_tracker)
        df = dp.encode_one_hot(df,cf.one_hot_features)
        df_test = dp.encode_one_hot(df_test,cf.one_hot_features)

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
    if verbose:
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

        if verbose:
            print "\n\t{} features of type '{}' are present".format(len(f_type_indices[f_type]),f_type)

        if f_type in cf.feature_types_to_exclude_list:

            if verbose:
                print "\t\tExcluding feature type: '{}'".format(f_type)

            features_mask[f_type_indices[f_type]] = False

    if verbose:
        print "\n\tList of excluded individual features:\n\t{}"\
            .format([i for i in cf.features_to_exclude_list])

    if unrecognised_features:
        if verbose:
            print "\n\tList of unrecognised features:\n\t{}"\
            .format(unrecognised_features.keys())
    else:
        del unrecognised_features

    ML_inputs["feature_types"] = f_type_indices
    del f_type_indices

    #display the left features
    if verbose:
        print "\n\tTotal of {} features left in dataset".format(features_mask.sum())
    
    #creating syntetic control target
    control_weights = 2*np.random.rand(features_mask.sum())-1
    for cat,dataset in [("train",df),("test",df_test)]:
        n = dataset.shape[1]
        array = np.sum(dataset.as_matrix()[:,features_mask].astype(float) * control_weights,
                       axis = 1)
        dataset["control_targets"] = array
    del control_weights
        #dataset.to_csv("data/Jurkat_hiv_{}_50kb.txt".format(cat),sep="\t",index=False)        
    #return
    
    
    # preprocess features of specific types
    step_tracker += 1
    if verbose:
        print "\n\n{}. Preprocessing feature type values".format(step_tracker)
    preprocess_funcs = {}
    for f_type in ML_inputs["feature_types"]:
        if f_type in cf.feature_types_to_exclude_list:
            continue
        idx = ML_inputs["feature_types"][f_type]
        p_fun = f_types.feature_types[f_type]["preprocess"]
        if p_fun is not None:
            if verbose:
                print "\n\tPreprocessing '{}' features".format(f_type)

            if f_type in cf.prepros_params:
                p_fun = p_fun(ml_method = cf.ML_estimator,**cf.prepros_params[f_type])
            else:
                p_fun = p_fun(ml_method = cf.ML_estimator)

            df.iloc[:,idx] = p_fun.fit_transform(df.iloc[:,idx].values)
            df_test.iloc[:,idx] = p_fun.transform(df_test.iloc[:,idx].values)
            nan_samples = (np.sum(np.isnan(df.iloc[:,idx].values),axis = 1) > 0).sum()
            test_nan_samples = (np.sum(np.isnan(df_test.iloc[:,idx].values),axis = 1) > 0).sum()
            if verbose:
                print "\t{} train and {} test  samples remaining with Nan values for '{}' features."\
                .format(nan_samples,test_nan_samples,f_type)
        preprocess_funcs[f_type] = p_fun
    del idx,p_fun,nan_samples,test_nan_samples


    
    # get target values from the dataset
    step_tracker += 1
    if verbose:
        print "\n\n{}. Extracting target values".format(step_tracker)
        print "\n\tUsing '{}' target type".format(cf.target_type["name"])
    targets_p_fun = t_types.target_types[cf.target_type["name"]](**cf.target_type["params"])
                                                      
    train_targets = targets_p_fun.fit_transform(df)
    nan_targets = np.isnan(train_targets).ravel()

    test_targets = targets_p_fun.transform(df_test)
    test_nan_targets = np.isnan(test_targets).ravel()
    if verbose:
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

    if verbose:
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
    if verbose:
        print "\n\n{}. Exctacting feature matrix".format(step_tracker)
    train_samples = df.as_matrix()
    ML_inputs["train_samples"] = train_samples

    test_samples = df_test.as_matrix()
    ML_inputs["test_samples"] = test_samples

    del train_samples,test_samples
    

    #Masking the datasets
    train_samples_mask = np.array(train_samples_mask)
    test_samples_mask = np.array(test_samples_mask)
    features_mask = np.array(features_mask)
    
    step_tracker += 1
    if verbose:
        print "\n\n{}. Masking the ML_inputs matrices".format(step_tracker)

    ML_inputs["feature_names"] = ML_inputs["feature_names"][features_mask]
    #for i,feature_name in enumerate(ML_inputs["feature_names"]):
     #   ML_inputs
    for f_type in ML_inputs["feature_types"].keys():
        indices = ML_inputs["feature_types"][f_type]
        updated_indices = np.zeros_like(features_mask,dtype = bool)
        updated_indices[indices] = True
        updated_indices = updated_indices[features_mask]
        ML_inputs["feature_types"][f_type] = np.where(updated_indices)
        
        
    
    ML_inputs["train_samples"] = ML_inputs["train_samples"]\
                                 [train_samples_mask,:][:,features_mask].astype(np.float)
    
    ML_inputs["test_samples"] = ML_inputs["test_samples"]\
                                 [test_samples_mask,:][:,features_mask].astype(np.float)

    ML_inputs["train_targets"] = ML_inputs["train_targets"][train_samples_mask].astype(np.float)
    ML_inputs["test_targets"] = ML_inputs["test_targets"][test_samples_mask].astype(np.float)

    if verbose:
        print "\n\t Final train dataset shape: {}".format(ML_inputs["train_samples"].shape)
        print "\t Final test dataset shape: {}".format(ML_inputs["test_samples"].shape)
    '''
    ML_inputs["mask"] = namedtuple("ML_inputs_mask","train_samples_mask,test_samples_mask,features_mask")(**{
        
        "train_samples_mask":np.array(train_samples_mask),
        "test_samples_mask": np.array(test_samples_mask),                              
        "features_mask" : np.array(features_mask),
        
        })
    '''
    if cf.sample_weights:
        step_tracker += 1
        if verbose:
            print "\n\n{}. Including sample weights".format(step_tracker)
        ML_inputs["train_sample_weight"] = aux.get_sample_weights(df,train_samples_mask)
        ML_inputs["test_sample_weight"] = aux.get_sample_weights(df_test,test_samples_mask)


    ML_inputs["preprocessing"] = preprocess_funcs

    #convert inputs to tuple
    ML_inputs = ML_inputs_tuple(ML_inputs)
    
    return ML_inputs

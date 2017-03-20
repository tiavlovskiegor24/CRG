### Control file stores all the required lists and parameters for the automatic machine learning pipeline execution

### Select machine learning method to use ###
'''
    valid options:
    "SVM" - Support Vector Machines 
    "RF_reg"  - Random Forest Regression
    "Rf_class" - Random Forest Classifier

'''

ml_method = "SVM"

### Specify feature types to exclude from ML training ###
feature_types_to_exclude_list = {
    "chip_c_hb" : None,
    "chip_c_zb" : None,
    "categorical": None,
}

### Specify individual features to exclude from ML training ###
features_to_exclude_list = {
    # list of certain features to exlude
    'brcd':None,
    'pos':None,
    'gene_name':None,
    "rep":None,
    "expr":None,
    "nread":None,
    "mapq":None,
    "chrom":None,
    "RNA":None,
    "DNA":None,

    # possible features remaining from older versions of datasets 
    "pos_expr":None,
    "targets":None,
    
}


### Specify sample groups to index by ###

sample_groups = ["chrom"]

### Specify Dataset location ###
source = "data/Jurkat_hiv_{}_50kb.txt"


### Indicate target feature selection and preprocessing routine ###

def get_targets(dataset,in_dataset = True):

    # takes the column 'targets' from dataset if in_dataset is true
    if in_dataset:
        array = dataset["targets"].values.reshape(-1,1)
        return array

    # otherwise targets are computed from scratch
    
    import numpy as np
    # currently target values are assumed
    print "\tComputing the RNA/DNA expression ratio as our target values"
    exp_ratio = (dataset["RNA"]*1.0/dataset["DNA"]).values

    # choose if 
    binary = False
    if binary:
        threshold = 3
        print "\tTransforming the problem into binary classification"
        print "\tSetting targets with expression ratio >= {} to 1, else 0".format(threshold)
        array = np.where(exp_ratio.values >= threshold,1.,0.).reshape(-1,1).astype(np.float)
    
        print "\tBinary label split: %.2f"%(array.sum()/array.shape)[0]
        #print y.shape

    else:
        print "Problem is a regression with targets on a continuous scale"
        print "\tTaking the log of targets (expression ratio)"
        array = np.log1p(exp_ratio).reshape(-1,1)

        print "\tRescaling the targets to 0-1 range"
        array = (array-np.nanmin(array,axis = 0,keepdims = True))\
                /(np.nanmax(array,axis = 0,keepdims = True)-np.nanmin(array,axis = 0,keepdims = True))
        
    return array
    




### Control file stores all the required lists and parameters for the automatic machine learning pipeline execution

### Select machine learning method to use ###
'''
    valid options:
    "SVM" - Support Vector Machines 
    "RF_reg"  - Random Forest Regression
    "Rf_class" - Random Forest Classifier

'''

ML_estimator = "Log_C"

### Specify feature types to exclude from ML training ###
feature_types_to_exclude_list = {
    "chip_c_hb" : None,
    #"chip_c_hb_r" : None,
    "chip_c_zb" : None,
    #"chip_c_zb_r" : None,
    "categorical": None,
    #"distance":None,
    
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
    "strand_oh__+":None,
    "cat_oh__SG":None,

    # possible features remaining from older versions of datasets 
    "pos_expr":None,
    "targets":None,
    
}


### Specify sample groups to index by ###

sample_groups = ["chrom"]

### Specify Dataset location ###
source = "data/Jurkat_hiv_{}_50kb.txt"


### Indicate target type and pointer to its selection and preprocessing object with any params ###

target_type = {
    "name":"test_targets",
    "params":{
        "source_target_type":"exp_ratio_cont",      
    },
}
'''
target_type = {
    "name":"exp_ratio_bin",
    "params":{"threshold":3},
}
target_type = {
    "name":"exp_ratio_cont",
    "params":{},
}

target_type = {
    "name":"test_targets",
    "params":{},
}

target_type = {
    "name":"exp_ratio_multiclass",
    "params":{
        "classes" : {
            "low_exp_ratio" : ("x < 3",0),
            "medium_exp_ratio" :("(x >= 3) & (x < 5)",1),
            "high_exp_ratio" : ("x >= 5",2),
        },
    },
}
'''


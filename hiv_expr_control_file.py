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
    "chip_c_hb",
    "chip_c_hb_r",
    "chip_c_zb",
    "chip_c_zb_r",
    "categorical",
    "distance",
    
}

### Specify individual features to exclude from ML training ###
features_to_exclude_list = {
    # list of certain features to exlude
    'brcd',
    'pos',
    'gene_name',
    "rep",
    "expr",
    "nread",
    "mapq",
    "chrom",
    "RNA",
    "DNA",
    "strand_oh__+",
    "cat_oh__SG",
    'exprscore', 'reads', 'dna', 'rna', 'locus',

    # possible features remaining from older versions of datasets 
    "pos_expr",
    "targets",
    "strand",
    "cat",
    "control_targets",
    
}

### set of categorical features to encode using one-hot encoding
one_hot_features = {
    #"cat",
    "strand",
}


### Specify sample groups to index by ###

sample_groups = ["chrom"]

### Include whether to consider sample weights in optimisation ####
sample_weights = False

### Specify Dataset location ###
source = "data/Jurkat_hiv_{}_50kb.txt"


### Preprocessing parameters ###
prepros_params = {

    "gmfpt":{
        "skip":False,
        "tail_compaction":(2.,98.)
    },

    "distance":{
        "skip":False,
    },
}

### Indicate target type and pointer to its selection and preprocessing object with any params ###

target_type = {
    "name":"expr_score_bin",
    "params":{"threshold":0.1},
}



'''
target_type = {
    "name":"in_dataset",
    "params":{
        "column_name":"exprscore",
        "tail_compaction":None,
        "scale":False,
        "nan_values":None,
        "log_values":False,
    },
}

target_type = {
    "name":"test_targets",
    "params":{
        "source_target_type":"exp_ratio_cont",      
    },
}

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


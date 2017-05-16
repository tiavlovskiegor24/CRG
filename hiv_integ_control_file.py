### Control file stores all the required lists and parameters for the automatic machine learning pipeline execution

### Select machine learning method to use ###
# see the full list of available estimators in ml_estimators.py file

ML_estimator = "Lasso_R"

### Specify feature types to exclude from ML training ###
feature_types_to_exclude_list = {
    "chip_c_hb",
    #"chip_c_hb_r",
    "chip_c_zb",
    #"chip_c_zb_r",
    "categorical",
    "distance",
}

### Specify individual features to exclude from ML training ###
features_to_exclude_list = {
    "bin",
    "chrom",
    "integ_density",
    #"control_targets",
    #"gmfpt",
}


### Specify sample groups to index by ###

sample_groups = ["chrom"]

#### Select a particlular group of samples from the full dataset ####
select_sample_group = {
    "chrom":"x == 'chr19'",
    "integ_density":"x > 0",
}

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



### Include whether to consider sample weights in optimisation ####
sample_weights = False

### Specify Dataset location ###
source = "data/hiv_integration_{}_50kb.txt"


### Indicate target type and pointer to its selection and preprocessing object with any params ###

target_type = {
    "name":"in_dataset",
    "params":{
        "column_name":"integ_density",
        "tail_compaction":None,
        "scale":False,
        "nan_values":0,
        "log_values":False,
    },
}


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
}

### Specify individual features to exclude from ML training ###
features_to_exclude_list = {
    'brcd':None,
    'pos':None,
    'gene_name':None,
    "rep":None,
    "expr":None,
    "nread":None,
    "mapq":None,
    "chrom":None,
}


### Specify sample groups to index by ###

sample_groups = ["chrom"]

### Specify Dataset location ###
source = "data/Jurkat_hiv_train_50kb.txt"

### Indicate target feature name ###
target = ""


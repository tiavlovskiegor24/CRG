### Control file stores all the required lists and parameters for the automatic machine learning pipeline execution

### 1. Select machine learning method to use ###
'''
    valid options:
    "SVM" - Support Vector Machines 
    "RF_reg"  - Random Forest Regression
    "Rf_class" - Random Forest Classifier

'''

ml_method = "SVM"

### 2. Specify features to drop ###
features_to_drop_list = [
    'brcd',
    'pos',
    'gene_name',
    "rep",
    "expr",
    "nread",
    "mapq",
    "chrom",
]


### 3. Specify sample groups to index by ###

sample_groups = ["chrom"]

### 4. Specify Dataset location ###
source = "data/Jurkat_hiv_train_50kb.txt"

### 4. Indicate target feature name ###
target = ""

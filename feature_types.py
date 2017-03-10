
# Look up table of existing feature_types

### FOLLOW THE STEPS BELOW

### 1. Import the relevant modules for preprocessing ###
import numpy as np



### 2. Include definitions of preprocessing functions if necessary ###

def distance_preprocessing(array,ml_method = None):
    # assuming feature values are stored as columns

    #Nan handling
    print "\tSubstituting NaNs with the values 10 times larger than max value found in the dataset"
    max_value = np.nanmax(array)
    print "\tMax value in the data{}".format(max_value)
    array = np.where(np.isnan(array),max_value*10,array)

    if ml_method == "SVM":
        print "Applying log1p to distance values"
        array = np.log1p(array)
        
    return array




### 3. Add the feature type entry to the dictionary below with the following default format ###

'''
    "feature_type_name" : {

        "id_fun":(lambda x: True if (???) else False), # function that takes the string name \
        #of the feature and return True if feature belongs to this feature type

        "preprocess":reduce((lambda x,fun: fun(x) ),[fun1,fun2,...]), # a list of \
        #preprocessing functions [fun1,fun2,...] to be applied on the feature values

        "file_format":".txt", # details of the feature file format 

        "about":"...", # short description of the feature type
    }, # don't forget a comma at the end of the entry
'''



feature_types_dict = {

    # genomic distance feature type
    "distance" : {

        "id_fun":(lambda x:True if ( x[:2]=="d_" or x.find("_d_") > -1 ) else False),

        "preprocess":distance_preprocessing,

        "file_format":"",

        "about":"distance in bases from particular locus"
    },


    # Global Mean First Passage Time

    "gmfpt" : {

        "id_fun":(lambda x: True if (x.find("gmfpt") > -1) else False), # function that takes \
        #the string name of the feature and return True if feature belongs to this feature type

        "preprocess":None, # include list of \
        #preprocessing functions [fun1,fun2,...] to be applied on the feature values

        "file_format":"tab separated bed file with in .txt format", # details of the \
        #feature file format 

        "about":"Computed Global Mean First Passage Time values of Hi-C matrix treated as a graph" \
        # short description of the feature type
    },


    # features created from categorical ones using one_hot encoding
    "one_hot" : {
    
        "id_fun" : (lambda x: True if (x.find("_oh_") > -1) else False), 

        "preprocess" : None, 
        
        "file_format" : "this features are not stored explicitly in files", 
       
        "about" : "features created from categorical ones using one_hot encoding", 
        
    }, # don't forget a comma at the end of the entry
}

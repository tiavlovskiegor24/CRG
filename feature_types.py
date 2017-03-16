
# Look up table of existing feature_types

### FOLLOW THE STEPS BELOW


### 1. Include definitions of preprocessing functions if necessary ###



def distance_preprocessing(array,ml_method = None,skip = False):

    if skip:
        print "\tSkipping the preprocessing of 'distance' features"
        return array
    
    import numpy as np
    # assuming feature values are stored as columns

    #Nan handling
    print "\tSubstituting NaNs with the values 10 times larger than max value found in the dataset"
    max_value = np.nanmax(array)
    print "\tMax value in the data {}".format(max_value)
    array = np.where(np.isnan(array),max_value*10,array)

    if ml_method not in ["RF"]:
        print "\tApplying log1p to distance values"
        array = np.log1p(array)

        print "\tRescaling 'distance' features to 0-1 range"
        array = (array-np.nanmin(array,axis = 0,keepdims = True))\
                /(np.nanmax(array,axis = 0,keepdims = True)-np.nanmin(array,axis = 0,keepdims = True))

        
    return array


########################### Hi-C features ########################################################

def gmfpt_preprocessing(array,ml_method = None):
    import numpy as np
    
    #Nan handling
    #Nans typically lie in non-reachable region and thus it is better to remove this samples
    #Leave Nans and get_ML_inputs will take care of them

    if ml_method not in []:
        print "\tApplying log1p to 'gmfpt' values"
        array = np.log1p(array)

        # removing outliers at the tails
        percent = 1.
        print "\tRemoving top and bottom {}% percent of samples".format(percent)
        upper = np.nanpercentile(array,100-percent,axis = 0,keepdims = True)
        lower = np.nanpercentile(array,percent,axis = 0,keepdims = True)
        nan_mask = np.where(~np.isnan(array))
        array[nan_mask] = np.where(
            np.greater_equal(array[nan_mask],lower) &\
            np.less_equal(array[nan_mask],upper)
            ,array[nan_mask],np.nan)

        print "\tRescaling 'gmfpt' to 0-1 range"
        array = (array-np.nanmin(array,axis = 0,keepdims = True))\
                /(np.nanmax(array,axis = 0,keepdims = True)-np.nanmin(array,axis = 0,keepdims = True))

    return array



### contact decay feature ####
def contact_decay_preprocessing(array,ml_method,skip = False):

    if skip:
        print "\tSkipping the preprocessing of 'contact_decay' features"
        return array

    
    import numpy as np

    #Nan handling
    #Nans typically lie in non-reachable region and thus it is better to remove this samples
    #Leave Nans and get_ML_inputs will take care of them
 
    if ml_method not in []:
        # removing outliers at the tails
        percent = 1.
        print "\tRemoving top and bottom {}% percent of samples".format(percent)
        upper = np.nanpercentile(array,100-percent,axis = 0,keepdims = True)
        lower = np.nanpercentile(array,percent,axis = 0,keepdims = True)
        nan_mask = np.where(~np.isnan(array))
        array[nan_mask] = np.where(
            np.greater_equal(array[nan_mask],lower) &\
            np.less_equal(array[nan_mask],upper)
            ,array[nan_mask],np.nan)

        print "\tRescaling 'contact_decay' to 0-1 range"
        array = (array-np.nanmin(array,axis = 0,keepdims = True))\
                /(np.nanmax(array,axis = 0,keepdims = True)-np.nanmin(array,axis = 0,keepdims = True))


    return array

contact_decay = {

    "id_fun":(lambda x: True if (x.find("c_decay") > -1) or \
              (x.find("contact_decay") > -1)  else False),

    "preprocess" : contact_decay_preprocessing, 
    
    "file_format" : "bed file", 
       
    "about" : "exponential decay constant of the contact counts for each locus", 

}


### row sum feature ####
def row_sum_preprocessing(array,ml_method):
    import numpy as np
    #Nan handling
    #Nans typically lie in non-reachable region and thus it is better to remove this samples
    #Leave Nans and get_ML_inputs will take care of them

    if ml_method not in []:

        print "\tApplying log1p to 'row_sum' values"
        array = np.log1p(array)

        # removing outliers at the tails
        percent = 1.
        print "\tRemoving top and bottom {}% percent of samples".format(percent)
        upper = np.nanpercentile(array,100-percent,axis = 0,keepdims = True)
        lower = np.nanpercentile(array,percent,axis = 0,keepdims = True)
        nan_mask = np.where(~np.isnan(array))
        array[nan_mask] = np.where(
            np.greater_equal(array[nan_mask],lower) &\
            np.less_equal(array[nan_mask],upper)
            ,array[nan_mask],np.nan)        

        print "\tRescaling 'row_sum' to 0-1 range"
        array = (array-np.nanmin(array,axis = 0,keepdims = True))\
                /(np.nanmax(array,axis = 0,keepdims = True)-np.nanmin(array,axis = 0,keepdims = True))

    return array

row_sum = {

    "id_fun":(lambda x: True if (x.find("row_sum") > -1) else False),

    "preprocess" : row_sum_preprocessing, 
    
    "file_format" : "bed file", 
       
    "about" : "sum of Hi-C elements row wise", 
}


def intra_inter_ratio_preprocessing(array,ml_method,skip = False):

    if skip:
        print "\tSkipping the preprocessing of 'intra_inter_ratio' features"
        return array

    
    import numpy as np

    #Nan handling
    #Nans typically lie in non-reachable region and thus it is better to remove this samples
    #Leave Nans and get_ML_inputs will take care of them
 
    if ml_method not in []:
        # removing outliers at the tails
        percent = 1.
        print "\tRemoving top and bottom {}% percent of samples".format(percent)
        upper = np.nanpercentile(array,100-percent,axis = 0,keepdims = True)
        lower = np.nanpercentile(array,percent,axis = 0,keepdims = True)
        nan_mask = np.where(~np.isnan(array))
        array[nan_mask] = np.where(
            np.greater_equal(array[nan_mask],lower) &\
            np.less_equal(array[nan_mask],upper)
            ,array[nan_mask],np.nan)

        print "\tRescaling 'intra_inter_ratio' to 0-1 range"
        array = (array-np.nanmin(array,axis = 0,keepdims = True))\
                /(np.nanmax(array,axis = 0,keepdims = True)-np.nanmin(array,axis = 0,keepdims = True))


    return array

intra_inter_ratio = {

    "id_fun":(lambda x: True if (x.find("intra_inter_ratio") > -1) else False),

    "preprocess" : intra_inter_ratio_preprocessing, 
    
    "file_format" : "bed file", 
       
    "about" : "ratio of intra to inter contacts of Hi-C data", 

}


### ab_score ######

def ab_score_preprocessing(array,ml_method,skip = False):

    if skip:
        print "\tSkipping the preprocessing of 'ab_score' features"
        return array

    
    import numpy as np

    #Nan handling
    #Nans typically lie in non-reachable region and thus it is better to remove this samples
    #Leave Nans and get_ML_inputs will take care of them
 
    if ml_method not in []:

        print "\tRescaling 'ab_score' to 0-1 range"
        array = (array+100)/200.
                


    return array

ab_score = {

    "id_fun":(lambda x: True if (x.find("ab_score") > -1) else False),

    "preprocess" : ab_score_preprocessing, 
    
    "file_format" : "bed file", 
       
    "about" : "ab_score of each locus computed by Eduard", 

}



    

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

        "preprocess":gmfpt_preprocessing, # include list of \
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


    # Chip-C features created by Eduard
    "chip_c_zb" : {

        "id_fun" : (lambda x: True if (x[-3:] == "_zb")  else False), 

        "preprocess" :None, 
        
        "file_format" : "this features are not stored explicitly in files", 
       
        "about" : "Total count of enriched (p=1.0) zerone bins (300bp) within all contacts", 
        
    },


    "chip_c_zb_r" : {

        "id_fun" : (lambda x: True if (x[-5:] == "_zb_r")  else False), 

        "preprocess" : None, 
        
        "file_format" : "this features are not stored explicitly in files", 
       
        "about" : "Mean number of enriched zerone bins (300bp) per Hi-C bin (50kb)", 
        
    },

    
    "chip_c_hb" : {

        "id_fun" : (lambda x: True if (x[-3:] == "_hb")  else False), 

        "preprocess" : None, 
        
        "file_format" : "this features are not stored explicitly in files", 
       
        "about" : "Number of Hi-C contacts (50kb) containing at least one enriched zerone bin (300bp).", 
        
    },


    "chip_c_hb_r" : {

        "id_fun" : (lambda x: True if (x[-5:] == "_hb_r")  else False), 

        "preprocess" : None, 
        
        "file_format" : "this features are not stored explicitly in files", 
       
        "about" : "rate of Hi-C contacts containing at least one enriched zerone bin. (i.e. hb/#contacts)", 
        
    },


    # categorical features
    "categorical" : {

        "id_fun" : (lambda x: True if (x.find("categ_") > -1) else False), 

        "preprocess" : None, 
        
        "file_format" : "", 
       
        "about" : "categorical features", 
        
    },

    
    #row_sum feature
    "row_sum" : row_sum,

    #contact decay featue
    "contact_decay" : contact_decay,

    #intra inter ratio feature
     "intra_inter_ratio" : intra_inter_ratio,

    #ab_score
    "ab_score" : ab_score,
    
    
}
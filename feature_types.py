
# Look up table of existing feature_types

### FOLLOW THE STEPS BELOW


### 1. Include definitions of preprocessing functions if necessary ###

### parent class for preprocessing values of particular type
class Feature_Type(object):

    # by default feature values are not changed
    def __init__(self,feature_type = None,
                 skip=False,
                 handle_nans = None,
                 log_values = None,
                 shrink_tails = None,
                 rescale_values = None
    ):
        self.ml_method = None # this is irrelevant attribute necessary for old preprocessing
        self.feature_type = feature_type
        self.skip = skip
        self.handle_nans = handle_nans
        self.log_values = log_values
        self.shrink_tails = shrink_tails
        self.rescale_values = rescale_values

        # this the default order in which preprocessing
        # functions will be applied
        # order can be changes when creating child classes
        # and new preprocess functions can be added to the list
        self.process_order = [
            self.handle_nans_fun,
            self.log_values_fun,
            self.shrink_tails_fun,
            self.rescale_values_fun,
        ]

    def handle_nans_fun(self,array,mode = "fit_transform",**kwargs):

        if self.handle_nans not in [None,False]:
            sub_value = self.handle_nans
            if mode == "fit_transform":
                print "\tSubstituting Nan values with '{}'".format(sub_value)
            array = np.where(nan_indices,sub_value,array)
            
            
        return array
                

    def log_values_fun(self,array,mode = "fit_transform",**kwargs):

        if self.log_values not in [None,False]:
            if mode == "fit_transform":
                print "\t Applying log1p to '{}' values".format(self.feature_type)
            array = np.log1p(array)
            
        return array


    def shrink_tails_fun(self,array,mode = "fit_transform",**kwargs):

        if self.shrink_tails not in [None,False]:
            if mode == "fit_transform":
                self.lower_percentile,self.upper_percentile = self.shrink_tails
                print "\tShrinking top {}% and bottom {}% of samples"\
                    .format(self.upper_percentile,self.lower_percentile)

                array = aux.linear_tail_compaction(self,array,fit = True)
            elif mode == "transform":
                array = aux.linear_tail_compaction(self,array,fit = False)

        return array


    def rescale_values_fun(self,array,mode = "fit_transform",**kwargs):

        if self.rescale_values not in [None,False]:
            if mode == "fit_transform":
                print "\tRescaling values to 0-1 range using 'max-min'"
                self.min_values = np.nanmin(array,axis = 0,keepdims = True)
                self.max_values = np.nanmax(array,axis = 0,keepdims = True)

            array = (array-self.min_values)*1./(self.max_values-self.min_values)

            assert all(np.nansum(array,axis = 0) > 0),"features with all zero values present"

        return array
            
        
    def fit_transform(self,array):
        if self.skip:
            print "\tSkipping the preprocessing of '{}' features".format(self.feature_type)
            return array
            
        for process_fun in self.process_order:
            array = process_fun(array,mode = "fit_transform")

        return array

    def transform(self,array):

        if self.skip:
            return array
        
        for process_fun in self.process_order:
            array = process_fun(array,mode = "transform")

        return array


class distance_preprocessing(Feature_Type):

    def __init__(self,feature_type = "distance",**kwargs):
        super(distance_preprocessing,self).__init__(**kwargs)

    def fit_transform(self,array,skip = False):
        #method for extracting scaling parameters from train set and tranforming
        # the train set
        if not hasattr(self,'skip'):
            self.skip = skip
        
        if self.skip: 
            print "\tSkipping the preprocessing of 'distance' features"
            return array
        
        import numpy as np
        # assuming feature values are stored as columns

        #Nan handling
        print "\tSubstituting NaNs with the values 10 times larger than max value found in the dataset"
        self.max_value = np.nanmax(array)
        print "\tMax value in the data {}".format(self.max_value)
        self.max_value *= 10
        array = np.where(np.isnan(array),self.max_value,array)

        if self.ml_method not in ["RF"]:
            print "\tApplying log1p to 'distance' values"
            array = np.log1p(array)

            self.max_vals = np.nanmax(array,axis = 0,keepdims = True)
            self.min_vals = np.nanmin(array,axis = 0,keepdims = True)

            print "\tRescaling 'distance' features to 0-1 range"
            array = (array-self.min_vals)/(self.max_vals-self.min_vals)
                    
        return array

    def transform(self,array):

        if self.skip:
            
            return array

        import numpy as np
        #Nan handling
        
        max_value = np.nanmax(array)
        
        max_value *= 10
        array = np.where(np.isnan(array),max(self.max_value,max_value),array)

        if self.ml_method not in ["RF"]:
            
            array = np.log1p(array)

            array = (array-self.min_vals)/(self.max_vals-self.min_vals)
                    
        return array

distance = {

        "id_fun":(lambda x:True if ( x[:2]=="d_" or x.find("_d_") > -1 ) else False),

        "preprocess":distance_preprocessing,

        "file_format":"",

        "about":"distance in bases from particular locus"
    }



########################### Hi-C features ########################################################
import auxiliary_items as aux
import numpy as np

class gmfpt_preprocessing(Feature_Type):

    def __init__(self,**kwargs):

        super(gmfpt_preprocessing,self).__init__(**kwargs)

    def fit_transform(self,array):
        #method for extracting scaling parameters from train set and tranforming
        # the train set
        
        if self.skip: 
            print "\tSkipping the preprocessing of 'gmfpt' features"
            return array
        

        #Nan handling
        #Nans typically lie in non-reachable region and thus it is better to remove this samples
        #Leave Nans and get_ML_inputs will take care of them

        if self.ml_method not in []:
            m,n = array.shape
            
            print "\tApplying log1p to 'gmfpt' values"
            array = np.log1p(array)
            
            # processing outliers at the tails

            if not hasattr(self,"lower_percentile"):
                self.lower_percentile = 2.
                self.upper_percentile = 98.
            
            #default scaling values
            self.min_value = np.nanmin(array,axis = 0,keepdims = True)
            self.max_values = np.nanmax(array,axis = 0,keepdims = True)
            
            
            # shrinking values in top and bottom tails
            print "\tShrinking top {}% and bottom {}% of samples"\
                .format(self.upper_percentile,self.lower_percentile)
            array = aux.linear_tail_compaction(self,array,fit = True)
            
            print "\tRescaling 'gmfpt' to 0-1 range"
            array = (array-self.min_value)/(self.max_value-self.min_value)
                    

        return array


    def transform(self,array):

        if self.skip:
            print "\tSkipping the preprocessing of 'feature_type' features"
            return array

        import numpy as np

        #Nan handling
        #Nans typically lie in non-reachable region and thus it is better to remove this samples
        #Leave Nans and get_ML_inputs will take care of them

        if self.ml_method not in []:
            print "\tApplying log1p to 'gmfpt' values"
            array = np.log1p(array)

            # shrinking values in top and bottom tails
            array = aux.linear_tail_compaction(self,array,fit = False)

            #rescaling values to 0-1 range
            array = (array-self.min_value)/(self.max_value-self.min_value)
                    

        return array


gmfpt = {

        "id_fun":(lambda x: True if (x.find("gmfpt") > -1) else False), # function that takes \
        #the string name of the feature and return True if feature belongs to this feature type

        "preprocess":gmfpt_preprocessing, # include list of \
        #preprocessing functions [fun1,fun2,...] to be applied on the feature values

        "file_format":"tab separated bed file with in .txt format", # details of the \
        #feature file format 

        "about":"Computed Global Mean First Passage Time values of Hi-C matrix treated as a graph" \
        # short description of the feature type
    }


### contact decay feature ####

class Contact_Decay(Feature_Type):
    
    def __init__(self,**kwargs):

        super(Contact_Decay,self).__init__(feature_type = "contact_decay",
                                     shrink_tails = (1.,97.),
                                     rescale_values = True,
                                     **kwargs)


contact_decay = {

    "id_fun":(lambda x: True if (x.find("c_decay") > -1) or \
              (x.find("contact_decay") > -1)  else False),

    "preprocess" : Contact_Decay,
    
    "file_format" : "bed file", 
       
    "about" : "exponential decay constant of the contact counts for each locus", 

}



### row sum feature ####
class Row_Sum(Feature_Type):
    
    def __init__(self,**kwargs):

        super(Row_Sum,self).__init__(feature_type = "row_sum",
                                     shrink_tails = (1.,99.),
                                     rescale_values = True,
                                     **kwargs)
    
row_sum = {

    "id_fun":(lambda x: True if (x.find("row_sum") > -1) else False),

    "preprocess" : Row_Sum, 
    
    "file_format" : "bed file", 
       
    "about" : "sum of Hi-C elements row wise", 
}


###### intra_inter_contact ratio
class intra_inter_ratio_preprocessing(Feature_Type):

    def __init__(self,**kwargs):

        super(intra_inter_ratio_preprocessing,self).__init__(**kwargs)


    def fit_transform(self,array):
        #method for extracting scaling parameters from train set and tranforming
        # the train set
        
        if self.skip: 
            print "\tSkipping the preprocessing of 'intra_inter_ratio_preprocessing' features"
            return array
        
        
        import numpy as np

        #Nan handling
        #Nans typically lie in non-reachable region and thus it is better to remove this samples
        #Leave Nans and get_ML_inputs will take care of them

        if self.ml_method not in []:

            # processing outliers at the tails
            if not hasattr(self,"lower_percentile"):
                self.lower_percentile = 1.
                self.upper_percentile = 99.

            #default scaling values
            self.min_value = np.nanmin(array,axis = 0,keepdims = True)
            self.max_values = np.nanmax(array,axis = 0,keepdims = True)
            
            
            # shrinking values in top and bottom tails
            print "\tShrinking top {}% and bottom {}% of samples"\
                .format(self.upper_percentile,self.lower_percentile)
            array = aux.linear_tail_compaction(self,array,fit = True)

            # rescaling values to 0-1 range
            print "\tRescaling values to 0-1 range"
            array = (array-self.min_value)/(self.max_value-self.min_value)


            '''
            # removing outliers at the tails
            percent = 1.
            print "\tRemoving top and bottom {}% percent of samples".format(percent)
            self.upper = np.nanpercentile(array,100-percent,axis = 0,keepdims = True)
            self.lower = np.nanpercentile(array,percent,axis = 0,keepdims = True)
            nan_mask = np.where(~np.isnan(array))
            array[nan_mask] = np.where(
                np.greater_equal(array[nan_mask],self.lower) &\
                np.less_equal(array[nan_mask],self.upper)
                ,array[nan_mask],np.nan)

            print "\tRescaling 'intra_inter_ratio' to 0-1 range"
            array = (array-self.lower)/(self.upper-self.lower)
            '''

        return array


    def transform(self,array):
        #transform the data for prediction
        if self.skip:
            print "\tSkipping the preprocessing of 'intra_inter_ratio' features"
            return array

        import numpy as np

        #Nan handling
        #Nans typically lie in non-reachable region and thus it is better to remove this samples
        #Leave Nans and get_ML_inputs will take care of them

        if self.ml_method not in []:
            
            # shrinking values in top and bottom tails
            array = aux.linear_tail_compaction(self,array,fit = False)

            # rescaling values to 0-1 range
            array = (array-self.min_value)/(self.max_value-self.min_value)

            '''
            # removing outliers at the tails
            nan_mask = np.where(~np.isnan(array))
            array[nan_mask] = np.where(
                np.greater_equal(array[nan_mask],self.lower) &\
                np.less_equal(array[nan_mask],self.upper)
                ,array[nan_mask],np.nan)

            print "\tRescaling 'intra_inter_ratio' to 0-1 range"
            array = (array-self.lower)/(self.upper-self.lower)
            '''
        return array


intra_inter_ratio = {

    "id_fun":(lambda x: True if (x.find("intra_inter_ratio") > -1) else False),

    "preprocess" : intra_inter_ratio_preprocessing, 
    
    "file_format" : "bed file", 
       
    "about" : "ratio of intra to inter contacts of Hi-C data", 

}


### ab_score ######

class AB_Score(Feature_Type):

    def __init__(self,**kwargs):
        super(AB_Score,self).__init__(feature_type = "ab_score",
                                      rescale_values = True,
                                      **kwargs)

    def rescale_values(self,array):
        print "\tRescaling 'ab_score' to 0-1 range"
        array = (array+100)/200.

        return array

ab_score = {

    "id_fun":(lambda x: True if (x.find("ab_score") > -1) else False),

    "preprocess" : AB_Score, 
    
    "file_format" : "bed file", 
       
    "about" : "ab_score of each locus computed by Eduard", 

}



#### Chip-C features preprocessing ####

class chip_c_zb_r_preprocessing(Feature_Type):

    def __init__(self,**kwargs):

        super(chip_c_zb_r_preprocessing,self).__init__(**kwargs)


    def fit_transform(self,array):
        #method for extracting scaling parameters from train set and tranforming
        # the train set itself
        
        if self.skip: 
            print "\tSkipping the preprocessing of 'chip_c_zb_r' features"
            return array
        
        
        import numpy as np

        #Nan handling
        #Nans typically lie in non-reachable region and thus it is better to remove this samples
        #Leave Nans and get_ML_inputs will take care of them

        if self.ml_method not in []:

            #default scaling values
            self.min_value = np.nanmin(array,axis = 0,keepdims = True)
            self.max_values = np.nanmax(array,axis = 0,keepdims = True)
            
            # processing outliers at the tails
            if not hasattr(self,"lower_percentile"):
                self.lower_percentile = 1.
                self.upper_percentile = 99.

            
            # shrinking values in top and bottom tails
            print "\tShrinking top {}% and bottom {}% of samples"\
                .format(self.upper_percentile,self.lower_percentile)
            array = aux.linear_tail_compaction(self,array,fit = True)

            # rescaling values to 0-1 range
            print "\tRescaling values to 0-1 range"
            array = (array-self.min_value)/(self.max_value-self.min_value)

            '''
            print "\tRescaling 'chip_c_zb_r' to 0-1 range"
            self.max_vals = np.nanmax(array,axis = 0,keepdims = True)
            self.min_vals = np.nanmin(array,axis = 0,keepdims = True)
            array = (array-self.min_vals)/(self.max_vals-self.min_vals)
            '''
        return array

    def transform(self,array):

        if self.skip:
            print "\tSkipping the preprocessing of 'chip_c_zb_r' features"
            return array

        import numpy as np

        #Nan handling
        #Nans typically lie in non-reachable region and thus it is better to remove this samples
        #Leave Nans and get_ML_inputs will take care of them

        if self.ml_method not in []:
            
            # shrinking values in top and bottom tails
            array = aux.linear_tail_compaction(self,array,fit = False)

            # rescaling values to 0-1 range
            array = (array-self.min_value)/(self.max_value-self.min_value)

            '''
            print "\tRescaling 'chip_c_zb_r' to 0-1 range"
            array = (array-self.min_vals)/(self.max_vals-self.min_vals)
            '''
        return array
        

chip_c_zb_r = {

        "id_fun" : (lambda x: True if (x[-5:] == "_zb_r")  else False), 

        "preprocess" : chip_c_zb_r_preprocessing, 
        
        "file_format" : "this features are not stored explicitly in files", 
       
        "about" : "Mean number of enriched zerone bins (300bp) per Hi-C bin (50kb)", 
        
    }

class chip_c_hb_r_preprocessing(Feature_Type):

    def __init__(self,**kwargs):

        super(chip_c_hb_r_preprocessing,self).__init__(**kwargs)

    
    def fit_transform(self,array,skip = False):
        #method for extracting scaling parameters from train set and tranforming
        # the train set itself

        self.skip = skip
        
        if self.skip: 
            print "\tSkipping the preprocessing of 'chip_c_hb_r' features"
            return array
        
        
        import numpy as np

        #Nan handling
        #Nans typically lie in non-reachable region and thus it is better to remove this samples
        #Leave Nans and get_ML_inputs will take care of them

        if self.ml_method not in []:

            # processing outliers at the tails
            if not hasattr(self,"lower_percentile"):
                self.lower_percentile = 1.
                self.upper_percentile = 99.

            #default scaling values
            self.min_value = np.nanmin(array,axis = 0,keepdims = True)
            self.max_values = np.nanmax(array,axis = 0,keepdims = True)
            
            
            # shrinking values in top and bottom tails
            print "\tShrinking top {}% and bottom {}% of values"\
                .format(self.upper_percentile,self.lower_percentile)
            array = aux.linear_tail_compaction(self,array,fit = True)

            # rescaling values to 0-1 range
            print "\tRescaling values to 0-1 range"
            array = (array-self.min_value)/(self.max_value-self.min_value)
        
            '''
            print "\tRescaling 'chip_c_hb_r' to 0-1 range"
            self.max_vals = np.nanmax(array,axis = 0,keepdims = True)
            self.min_vals = np.nanmin(array,axis = 0,keepdims = True)
            array = (array-self.min_vals)/(self.max_vals-self.min_vals)
            '''
            
        return array

    def transform(self,array):

        if self.skip:
            print "\tSkipping the preprocessing of 'chip_c_hb_r' features"
            return array

        import numpy as np

        #Nan handling
        #Nans typically lie in non-reachable region and thus it is better to remove this samples
        #Leave Nans and get_ML_inputs will take care of them

        if self.ml_method not in []:
            
            # shrinking values in top and bottom tails
            array = aux.linear_tail_compaction(self,array,fit = False)

            # rescaling values to 0-1 range
            array = (array-self.min_value)/(self.max_value-self.min_value)

            '''
            print "\tRescaling 'chip_c_hb_r' to 0-1 range"
            array = (array-self.min_vals)/(self.max_vals-self.min_vals)
            '''
        return array




chip_c_hb_r = {

        "id_fun" : (lambda x: True if (x[-5:] == "_hb_r")  else False), 

        "preprocess" : chip_c_hb_r_preprocessing, 
        
        "file_format" : "this features are not stored explicitly in files", 
       
        "about" : "rate of Hi-C contacts containing at least one enriched zerone bin. (i.e. hb/#contacts)", 
        
}

#### Chip_z25 features ############# 
class Chip_z25(Feature_Type):

    def __init__(self,**kwargs):

        super(Chip_z25,self).__init__(feature_type = "chip_z25",**kwargs)


chip_z25 = {

        "id_fun" : (lambda x: True if (x[-4:] == "_z25")  else False),

        "preprocess" : Chip_z25,
        
        "file_format" : "this features are stored in txt file", 
       
        "about" : "chip_z25 were created by Guillaume", 
        
}




    

### 3. Add the feature type entry to the dictionary below with the following default format ###

feature_types = {

    # genomic distance feature type
    "distance" : distance,


    # Global Mean First Passage Time

    "gmfpt" : gmfpt,


    # features created from categorical ones using one_hot encoding
    "one_hot" : {
    
        "id_fun" : (lambda x: True if (x.find("_oh_") > -1) else False), 

        "preprocess" : None, 
        
        "file_format" : "this features are not stored explicitly in files", 
       
        "about" : "features created from categorical ones using one_hot encoding", 
        
    }, # don't forget a comma at the end of the entry


    # Chip feature created Guillaume
    "chip_z25" : chip_z25,

    # Chip-C features created by Eduard
    "chip_c_zb" : {

        "id_fun" : (lambda x: True if (x[-3:] == "_zb")  else False), 

        "preprocess" :None, 
        
        "file_format" : "this features are not stored explicitly in files", 
       
        "about" : "Total count of enriched (p=1.0) zerone bins (300bp) within all contacts", 
        
    },


    "chip_c_zb_r" : chip_c_zb_r,

    
    "chip_c_hb" : {

        "id_fun" : (lambda x: True if (x[-3:] == "_hb")  else False), 

        "preprocess" : None, 
        
        "file_format" : "this features are not stored explicitly in files", 
       
        "about" : "Number of Hi-C contacts (50kb) containing at least one enriched zerone bin (300bp).", 
        
    },


    "chip_c_hb_r" : chip_c_hb_r,


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

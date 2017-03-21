# module of all personally defined auxiliary functions

from collections import namedtuple
import numpy as np


class ML_inputs_tuple(object):
    
    def __init__(self,ML_inputs_dict):
        self.data = namedtuple("ML_inputs",ML_inputs_dict.keys())(**ML_inputs_dict)
    
    def __getitem__(self,key):
        return getattr(self.data,key)

    def get_data(self,train_or_test = "train",mask = True):
        
        if not mask:
            confirm = raw_input("Are you sure you want to get non-masked data?(Y/n)\n Some samples may contain Nans and some columns may not be suitable for Machine Learning")
            if confirm == "Y":
                samples = getattr(self.data,"{}_samples".format(train_or_test))
                targets = getattr(self.data,"{}_targets".format(train_or_test))
                return samples,targets
            
        print "Returning masked '{}' data".format(train_or_test)
        masks = getattr(self.data,"mask")
        samples_mask = getattr(masks,"{}_samples_mask".format(train_or_test))
        features_mask = getattr(self.data,"features_mask")
        
        samples = getattr(self.data,"{}_samples".format(train_or_test))\
                  [samples_mask,:][:,features_mask].astype(np.float)

        targets = getattr(self.data,"{}_targets".format(train_or_test))\
                  [samples_mask].astype(np.float)

        return samples,targets
        
    def __repr__(self):
        return self.data.__repr__()




def linear_tail_compaction(array,p_object,fit = True):
    '''
    function to linearly scale values of balow lower and above upper percentile to the average
    scale if inner region
    '''
    
    m,n = array.shape
    inner_percentile_range = p_object.upper_percentile - p_object.lower_percentile
    
    lower_value = np.nanpercentile(array,p_object.lower_percentile,axis = 0,keepdims = True)
    upper_value = np.nanpercentile(array,p_object.upper_percentile,axis = 0,keepdims = True)
    inner_value_range = upper_value - lower_value

    
    
    
    normal_percent_range = inner_value_range*1. / inner_percentile_range
    

    if fit:
        p_object.lower_tail_scaling = (lower_value-np.nanmin(array,axis = 0))\
                                      /(normal_percent_range*p_object.lower_percentile)
    
        p_object.upper_tail_scaling = (np.nanmax(array,axis = 0)-upper_value)\
                                      /(normal_percent_range*(100-p_object.upper_percentile))


    nan_mask = np.where(~np.isnan(array))
    
    array[nan_mask] = np.where(np.greater_equal(array[nan_mask],lower_value), 
                               array[nan_mask],
                               lower_value - \
                               (lower_value-array[nan_mask])/p_object.lower_tail_scaling)
    
    array[nan_mask] = np.where(np.less_equal(array[nan_mask],upper_value), 
                               array[nan_mask],
                               upper_value + \
                               (array[nan_mask]-upper_value)/p_object.upper_tail_scaling)
    if fit:
        p_object.min_value = np.nanmin(array,axis = 0)
        p_object.max_value = np.nanmax(array,axis = 0)

    return array

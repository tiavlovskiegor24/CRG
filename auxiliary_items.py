# module of all personally defined auxiliary functions
import numpy as np

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
        p_object.lower_tail_scaling = (lower_value-np.nanmin(array,axis = 0,keepdims = True))\
                                      /(normal_percent_range*p_object.lower_percentile)

        # remove zeros
        p_object.lower_tail_scaling[p_object.lower_tail_scaling == 0] = 1.

        
        p_object.upper_tail_scaling = (np.nanmax(array,axis = 0,keepdims = True)-upper_value)\
                                      /(normal_percent_range*(100-p_object.upper_percentile))

        #remove zeros
        p_object.upper_tail_scaling[p_object.upper_tail_scaling == 0] = 1.

        
    #mask nan entries
    nan_mask = np.where(~np.isfinite(array))
    array[nan_mask] = 0

    # compact lower tail
    array = np.where(np.greater_equal(array,lower_value), 
                               array,
                               lower_value - \
                               (lower_value-array)/p_object.lower_tail_scaling)

    # compact upper tail
    array = np.where(np.less_equal(array,upper_value), 
                               array,
                               upper_value + \
                               (array-upper_value)/p_object.upper_tail_scaling)

    # reinsert nan entries
    array[nan_mask] = np.nan
    
    if fit:
        p_object.min_value = np.nanmin(array,axis = 0,keepdims = True)
        p_object.max_value = np.nanmax(array,axis = 0,keepdims = True)

    return array

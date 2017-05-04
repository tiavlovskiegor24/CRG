#### Targets selection and preprocessing types
import numpy as np
import auxiliary_items as aux

class in_dataset(object):
    '''
    target are already in the dataset 
    '''
    def __init__(self,column_name = "control_targets",tail_compaction = None):
        self.column_name = column_name
        if tail_compaction is not None:
            self.lower_percentile,self.upper_percentile = tail_compaction
        
    def fit_transform(self,dataset):
        # takes the column from dataset
        print "\n\tTaking '{}' column as targets".format(self.column_name)
        array = dataset[self.column_name].values.ravel()

        self.max_value = np.nanmax(array,axis = 0,keepdims = True)
        self.min_value = np.nanmin(array,axis = 0,keepdims = True)

        if hasattr(self,"upper_percentile") and hasattr(self,"lower_percentile"):
            # processing outliers at the tails

            # shrinking values in top and bottom tails
            print "\n\tShrinking top {}% and bottom {}% of samples"\
                .format(self.upper_percentile,self.lower_percentile)
            array = aux.linear_tail_compaction(array,self,fit = True)

        

        print "\n\tRescaling the targets to 0-1 range"
        #array = (array-self.min_value)/(self.max_value-self.min_value)
        
        return array


    def transform(self,dataset):
        array = dataset[self.column_name].values.ravel()
        
        if hasattr(self,"upper_percentile") and hasattr(self,"lower_percentile"):
            # shrinking values in top and bottom tails
            array = aux.linear_tail_compaction(array,self,fit = False)

        
        #array = (array-self.min_value)/(self.max_value-self.min_value)
        return array


class exp_ratio_cont(object):
    '''
    targets are continuous values equal to the ratio of RNA to DNA expression
    '''
    def __init__(self):
        pass
    
    def fit_transform(self,dataset):

        import numpy as np
        # currently target values are assumed
        print "\n\tComputing the RNA/DNA expression ratio as our target values"
        array = (dataset["RNA"]*1.0/dataset["DNA"]).values

        print "\n\tProblem is a regression with targets on a continuous scale"
        print "\n\tTaking the log of targets (expression ratio)"
        array = np.log1p(array)

        self.max_value = np.nanmax(array,axis = 0,keepdims = True)
        self.min_value = np.nanmin(array,axis = 0,keepdims = True)


        # processing outliers at the tails
        self.lower_percentile = 0.
        self.upper_percentile = 99.

        
        # shrinking values in top and bottom tails
        print "\n\tShrinking top {}% and bottom {}% of samples"\
            .format(self.upper_percentile,self.lower_percentile)
        array = aux.linear_tail_compaction(array,self,fit = True)

        
        print "\n\tRescaling the targets to 0-1 range"
        array = (array-self.min_value)/(self.max_value-self.min_value)

        return array

    def transform(self,dataset):
        import numpy as np

        #computing the expression ration
        exp_ratio = (dataset["RNA"]*1.0/dataset["DNA"]).values
        # taking log of expression ratio
        array = np.log1p(exp_ratio)

        # shrinking values in top and bottom tails
        array = aux.linear_tail_compaction(array,self,fit = False)

        
        # scaling values to 0-1 range
        array = (array-self.min_value)/(self.max_value-self.min_value)

        return array


class exp_ratio_multiclass(object):

    def __init__(self,classes = None,**kwargs):

        if classes is None:
            classes = {
                "all_samples" : ("np.isfinite(x)",1),
            }
        self.classes = classes
        
    def fit_transform(self,dataset):
        import numpy as np
        # currently target values are assumed
        print "\n\tComputing the RNA/DNA expression ratio as our target values"
        array = (dataset["RNA"]*1.0/dataset["DNA"]).values

        print "\tTransforming the problem into multiclass classification\n"


        class_masks = {}
        for class_name,(condition,label) in self.classes.iteritems():

            masking_fun = aux.create_masking_fun(condition) 
            class_masks[class_name] = (label,condition,masking_fun(array))

        for class_name,(label,condition,mask) in class_masks.iteritems():
            
            print "\t\t Class '{}': Setting {:.0f}% of train values with '{}' to label '{}'"\
                .format(class_name,mask.sum()*100./mask.shape[0],condition,label)

            array = np.where(mask,float(label),array)

        return array
        

    def transform(self,dataset):
        array = (dataset["RNA"]*1.0/dataset["DNA"]).values

        class_masks = {}
        for class_name,(condition,label) in self.classes.iteritems():

            masking_fun = aux.create_masking_fun(condition) 
            class_masks[class_name] = (label,condition,masking_fun(array))
        print "\n"
        for class_name,(label,condition,mask) in class_masks.iteritems():
            
            print "\t\t Class '{}': Setting {:.0f}% of test values with '{}' to label '{}'"\
                .format(class_name,mask.sum()*100./mask.shape[0],condition,label)

            array = np.where(mask,float(label),array)
                                                       

        return array
        



class exp_ratio_bin(object):
    def __init__(self,threshold = 3):
        self.threshold = threshold
    
    def fit_transform(self,dataset):

        import numpy as np
        # currently target values are assumed
        print "\n\tComputing the RNA/DNA expression ratio as our target values"
        exp_ratio = (dataset["RNA"]*1.0/dataset["DNA"]).values

        print "\tTransforming the problem into binary classification"
        print "\tSetting targets with expression ratio >= {} to 1, else 0".format(self.threshold)
        array = np.where(exp_ratio >= self.threshold,1.,0.).astype(np.float)

        print "\tBinary label split: %.2f (proportion of ones)"%(array.sum()/array.shape[0])

        return array

    def transform(self,dataset):
        import numpy as np

        #computing the expression ration
        exp_ratio = (dataset["RNA"]*1.0/dataset["DNA"]).values

        array = np.where(exp_ratio >= self.threshold,1.,0.).astype(np.float)

        return array

    

class test_targets(object):
    def __init__(self,features_mask,noise=(),fields={},**kwargs):
        self.f_mask = features_mask
        self.noise = noise
        self.fields = fields
    
    def fit_transform(self,dataset):

        #form targets array
        if self.fields:
            array = np.zeros(dataset.shape[0])
            for field in self.fields:
                if self.fields[field] is None:
                    self.fields[field]  = 2*np.random.rand()-1
                array = array + self.fields[field]*dataset[field].values.ravel()
        else:
            n = dataset.shape[1]
            self.weights = 2*np.random.rand(self.f_mask.sum())-1
            array = np.sum(dataset.as_matrix()[:,self.f_mask].astype(float) \
                           * self.weights,
                           axis = 1).ravel()
        
        # add noise
        if self.noise:
            print "\n\tAdding noise with mean,std: ",self.noise
            array = array + np.random.normal(self.noise[0],self.noise[1],size = array.shape[0])

        self.max_value = np.nanmax(array,axis = 0,keepdims = True)
        self.min_value = np.nanmin(array,axis = 0,keepdims = True)

        print "\n\tRescaling the targets to 0-1 range"
        array = (array-self.min_value)/(self.max_value-self.min_value)
        
        return array

    def transform(self,dataset):

        if self.fields:
            array = np.zeros(dataset.shape[0])
            for field,weight in self.fields.iteritems():
                array = array + weight*dataset[field].values.ravel()
        else:
            n = dataset.shape[1]
            array = np.sum(dataset.iloc[:,np.arange(n)[self.f_mask]].values \
                           * self.weights,
                           axis = 1).ravel()
        '''
        if self.noise:
        # add noise
            array = array + np.random.normal(self.noise[0],self.noise[1],size = array.shape[0])        
        '''
        
        array = (array-self.min_value)/(self.max_value-self.min_value)
        return array


    
class test_randomised_targets(object):
    def __init__(self,source_target_type,**kwargs):
        self.source_target_type = target_types[source_target_type](**kwargs)
        
    def fit_transform(self,dataset):
        array = self.source_target_type.fit_transform(dataset)
        shuffled_array = np.random.permutation(array)
        assert np.equal(np.sort(array),np.sort(shuffled_array)).all(),"shuffled targets are not identical"
        return shuffled_array

    def transform(self,dataset):

        array = self.source_target_type.fit_transform(dataset)
        shuffled_array = np.random.permutation(array)
        assert np.equal(np.sort(array),np.sort(shuffled_array)).all(),"shuffled targets are not identical"
        return shuffled_array



    
target_types = {
    "exp_ratio_cont":exp_ratio_cont,
    "in_dataset":in_dataset,
    "exp_ratio_bin":exp_ratio_bin,
    "test_targets":test_targets,
    "test_randomised_targets":test_randomised_targets,
    "exp_ratio_multiclass":exp_ratio_multiclass,
}

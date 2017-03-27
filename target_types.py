#### Targets selection and preprocessing types
import numpy as np
import auxiliary_items as aux

class in_dataset(object):
    '''
    target are already in the dataset 
    '''
    def __init__(self):
        pass
    
    def fit_transform(self,dataset,column_name = 'targets'):

        # takes the column  from dataset if in_dataset is true
        self.column_name = column_name
        array = dataset[self.column_name].values

        return array


    def transform(self,dataset,in_dataset = False):
        array = dataset[self.column_name]
        return array


class exp_ratio_cont(object):
    '''
    targets are continuous values equal to the ratio of RNA to DNA expression
    '''
    def __init__(self):
        pass
    
    def fit_transform(self,dataset,in_dataset = False):

        # takes the column 'targets' from dataset if in_dataset is true
        if in_dataset:
            array = dataset["targets"].values
            return array

        # otherwise targets are computed from scratch

        import numpy as np
        # currently target values are assumed
        print "\n\tComputing the RNA/DNA expression ratio as our target values"
        exp_ratio = (dataset["RNA"]*1.0/dataset["DNA"]).values

        print "\n\tProblem is a regression with targets on a continuous scale"
        print "\n\tTaking the log of targets (expression ratio)"
        array = np.log1p(exp_ratio)

        self.max_value = np.nanmax(array,axis = 0,keepdims = True)
        self.min_value = np.nanmin(array,axis = 0,keepdims = True)

        print "\n\tRescaling the targets to 0-1 range"
        array = (array-self.min_value)/(self.max_value-self.min_value)

        return array

    def transform(self,dataset,in_dataset = False):
        import numpy as np
        # takes the column 'targets' from dataset if in_dataset is true
        if in_dataset:
            array = dataset["targets"].values
            return array

        # otherwise targets are computed from scratch

        #computing the expression ration
        exp_ratio = (dataset["RNA"]*1.0/dataset["DNA"]).values
        # taking log of expression ratio
        array = np.log1p(exp_ratio)
        # scaling values to 0-1 range
        array = (array-self.min_value)/(self.max_value-self.min_value)

        return array


class exp_ratio_multiclass(object):

    def __init__(self,class_masks = None,**kwargs):

        if class_masks is None:
            class_masks = {
                "all_samples" : aux.create_class_mask("np.isfinite(x)",1),
            }
        self.class_masks = class_masks
        
    def fit_transform(self,dataset):
        import numpy as np
        # currently target values are assumed
        print "\n\tComputing the RNA/DNA expression ratio as our target values"
        array = (dataset["RNA"]*1.0/dataset["DNA"]).values

        print "\tTransforming the problem into multiclass classification"
        
        self.class_labels = {}
        no_label = np.ones_like(array,dtype=bool)
        for class_description,(condition,label) in self.class_masks.iteritems():
            print "\n\t\t Class '{}': Setting '{}' values to label {}".format(class_description,\
                                                                         condition,label)

            class_labelling = aux.create_class_labelling(condition,label)

            array,no_label = class_labelling(array,no_label)

            
            self.class_labels[label] = class_description
            
        for label,class_description in self.class_labels.iteritems():
            print "\n\tOveral {:.1f}% of train samples are class {}"\
                .format((array == label).sum()*100./array.shape[0],class_description)

        return array
        

    def transform(self,dataset):
        array = (dataset["RNA"]*1.0/dataset["DNA"]).values

        no_label = np.ones_like(array,dtype=bool)
        for class_description,(condition,label) in self.class_masks.iteritems():

            class_labelling = aux.create_class_labelling(condition,label)

            array,no_label = class_labelling(array,no_label)
            
            
        for label,class_description in self.class_labels.iteritems():
            n_labels = (array == label).sum()*100./array.shape[0]
                                                               
            print "\n\tOveral {:.1f}% of test samples are class {}".format(n_labels,class_description)
                                                               

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

    
target_types = {
    "exp_ratio_cont":exp_ratio_cont,
    "in_dataset":in_dataset,
    "exp_ratio_bin":exp_ratio_bin,
    "test_targets":test_targets,
    "exp_ratio_multiclass":exp_ratio_multiclass,
}

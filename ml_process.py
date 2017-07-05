from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


from time import time
import numpy as np
import pandas as pd

import myplot
import ml_estimators


def grid_search(clf,X,y,param_grid,**kwargs):
    if param_grid:
        print "\n\tPerforming parameter grid search on:\n\t{}".format(param_grid)

    t = time()
    #print kwargs
    clf = GridSearchCV(clf, param_grid = param_grid,n_jobs = -1,**kwargs)
    clf.fit(X,y)
    print "\n\t\tTime taken:{:.2f}".format(time()-t)
    if clf.best_params_:
        print "\n\t\tBest parameters:",clf.best_params_

    for i in ["train","test"]:
        results_mean = clf.cv_results_['mean_{}_score'.format(i)][clf.best_index_]
        results_std = clf.cv_results_['std_{}_score'.format(i)][clf.best_index_]
        print("\n\t\tCV '{}' score (mean +/- 2*sd): {:.2f} (+/- {:.2f})"\
              .format(i,results_mean,results_std * 2))

    
    return clf


def fit(cf,ML_inputs,estimator_name = None,param_grid = None,by_groups = None,fit_params = None,class_weight = None,**kwargs):

    if estimator_name is None:
        from control_file import ML_estimator as estimator_name
        
    estimator_name,estimator,default_param_grid = ml_estimators.get_estimator(estimator_name)
    if estimator is None:
        print "Canceling..."
        return None

    if param_grid is None or param_grid == "default":
        param_grid = default_param_grid
    
    X,y = ML_inputs.get_data("train")

    if estimator_name[-3:] != "_MC":
        assert X.max() <= 1 and X.min() >= 0 and np.max(y) <= 1 and np.min(y) >= 0,\
            "Data is not scaled to 0-1 range"

    scaler = StandardScaler()
    #X = scaler.fit_transform(X)
    #X = X-0.5
    #print X
    #print np.dot(X.T,X).diagonal()
    #print X.mean(axis=0)
    #print X.std(axis=0)
    
    if by_groups is None:

        print "\n\tRunning ML with '{}' ...".format(estimator_name)

        if estimator_name[-2:] == "_C" or estimator_name[-3:] == "_MC":
            clf = estimator(class_weight = class_weight,**kwargs)
        else:
            clf = estimator(**kwargs)
            

        #temporary giving weights to sampels
        if hasattr(cf,"sample_weights"):
            if cf.sample_weights:    
                fit_params = {
                    "sample_weight" : ML_inputs["train_sample_weight"],
                }
        
        
        
        clf = grid_search(clf,X,y,param_grid,fit_params = fit_params)
        clf.scaler = scaler

        print "\n\tTrain score is {:.2f}".format(clf.best_estimator_.score(X,y,sample_weight = \
                                                                           ML_inputs["train_sample_weight"]))

        if estimator_name[-2:] == "_C" or estimator_name[-3:] == "_MC":
            print metrics.classification_report(y,clf.predict(X))
        elif estimator_name[-2:] == "_R":            
            f,ax = myplot.myplot(y,clf.predict(X),style = ".",shape = (1,2),figsize = (14,7),sharey = True)
            ax[0].set_title("Train samples")
            ax[0].set_ylabel("Predicted")
            ax[0].hlines(y.mean(),0,1)
        
        X,y = ML_inputs.get_data("test")
        #X = clf.scaler.transform(X)
        #X = X-0.5
        #print np.dot(X.T,X).diagonal()
        
        
        print "\n\tTest score is {:.2f}".format(clf.best_estimator_.score(X,y,sample_weight = ML_inputs["test_sample_weight"]))

        if estimator_name[-2:] == "_C":
            print metrics.classification_report(y,clf.predict(X))
        elif estimator_name[-2:] == "_R":
            ax[1].plot(y,clf.predict(X),".")
            ax[1].set_title("Test samples")
            ax[1].hlines(y.mean(),0,1)
        
            for axis in ax: 
                axis.set_xlabel("Observed")
                axis.set(aspect='equal')
                axis.set_xlim(xmin = -.01,xmax = 1.01)
                axis.set_ylim(ymin = -.01,ymax = 1.01)
                axis.grid(True)
            f.tight_layout()

        return clf
    else:
        
        while by_groups not in ML_inputs["sample_groups"]:
            by_group = raw_input("Select group from:\n {}\n or enter 'c' to CANCEL: "\
                              .format(ML_inputs["sample_groups"].keys()))
            if by_groups == "c":
                print "Cancelled"
                return None

        group_clfs = {}
        for name,idx in ML_inputs["sample_groups"][by_groups].iteritems():
            print "\nRunning {} for '{g}' = '{n}'".format(estimator_name,g = by_groups,n = name)
            X_group = X[idx]
            y_group = y[idx]
            m = y_group.shape[0]
            print "Group data set has {} samples".format(m)
            if m < 10:
                continue
            group_clfs[name] = estimator(X_group,y_group,**kwargs)
            
        return group_clfs

def predict(clf,X,select = "all",):
    #X = clf.scaler.transform(X)
    return clf.best_estimator_.predict(X)
    
def evaluate(clf,ML_inputs,X,y,select = "all",estimator_name = None):

    X,y = ML_inputs.get_data(select)
    

    print "\n\t'{}' score is {:.2f}".format(estimator_name,clf.best_estimator_\
                                            .score(X,y,sample_weight = None))

    if estimator_name[-2:] == "_C":
        print metrics.classification_report(y,clf.predict(X))
    elif estimator_name[-2:] == "_R":
        f,axis = myplot.myplot(y,clf.predict(X),style = ".",figsize = (14,7))
        axis.plot(y,clf.predict(X),".")
        axis.set_title("All samples")
        axis.hlines(y.mean(),0,1)

        
        axis.set_xlabel("Observed")
        axis.set(aspect='equal')
        axis.set_xlim(xmin = -.01,xmax = 1.01)
        axis.set_ylim(ymin = -.01,ymax = 1.01)
        axis.grid(True)
        f.tight_layout()

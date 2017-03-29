
from sklearn import metrics
from sklearn.model_selection import GridSearchCV



from time import time
import numpy as np
import pandas as pd

from myplot import myplot
import ML_estimators


def grid_search(clf,X,y,parameters,**kwargs):
    if parameters:
        print "\n\tPerforming parameter grid search on:\n\t{}".format(parameters)

    t = time()
    clf = GridSearchCV(clf, parameters,n_jobs = -1,**kwargs)
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


def run_ML(ML_inputs,estimator_name = None,parameters = None,by_groups = None,**kwargs):

    if estimator_name is None:
        from control_file import ML_estimator as estimator_name
        
    estimator_name,estimator,default_parameters = ML_estimators.get_estimator(estimator_name)
    if estimator is None:
        print "Canceling..."
        return None

    if parameters is None or parameters == "default":
        parameters = default_parameters
    
    X,y = ML_inputs.get_data("train")

    if estimator_name[-3:] != "_MC":
        assert np.max(X) <= 1 and np.min(X) >= 0 and np.max(y) <= 1 and np.min(y) >= 0,\
            "Data is not scaled to 0-1 range"

    
    if by_groups is None:

        print "\n\tRunning ML with '{}' ...".format(estimator_name)

        clf = estimator(**kwargs)
        
        clf = grid_search(clf,X,y,parameters)

        print "\n\tTrain score is {:.2f}".format(clf.score(X,y))

        if estimator_name[-2:] == "_C" or estimator_name[-3:] == "_MC":
            print metrics.classification_report(y,clf.predict(X))
        elif estimator_name[-2:] == "_R":            
            f,ax = myplot(y,clf.predict(X),style = ".",shape = (1,2),figsize = (14,7),sharey = True)
            ax[0].set_title("Train samples")
            ax[0].set_ylabel("Predicted")
            ax[0].hlines(y.mean(),0,1)
        
        X,y = ML_inputs.get_data("test")
        print "\n\tTest score is {:.2f}".format(clf.score(X,y))

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
    

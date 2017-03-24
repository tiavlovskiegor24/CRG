from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression,Lasso

from sklearn.preprocessing import StandardScaler,MinMaxScaler,minmax_scale
#from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.pipeline import make_pipeline

from time import time
#from sklearn.decomposition import PCA, KernelPCA

import numpy as np
import pandas as pd

from myplot import myplot


def grid_search(clf,X,y,parameters,**kwargs):
    if parameters:
        print "\n\tPerforming parameter grid search..."
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

ML_methods = {
    "DT_R":(DecisionTreeRegressor,
                    {
                        "max_features" : ("sqrt","auto",),
                        "min_samples_split" : (3,10,30,100,300),
                    }),
    
    "DT_C":(DecisionTreeClassifier,
            {
                "max_features" : ("sqrt","auto",),
                "min_samples_split" : (3,10,30,100,300),
            }),

    "RF_C":(RandomForestClassifier,
            {
                "max_features" : ("sqrt","auto",),
                "min_samples_split" : (3,10,30,100,300),
                "n_estimators" : (10,20,30),            
            }),

    "RF_R":(RandomForestRegressor,
                    {
                        "max_features" : ("sqrt","auto",),
                        "min_samples_split" : (2,10,30,100,300),
                        "n_estimators" : (10,20,30),
                    }),

    "SV_C":(make_pipeline(MinMaxScaler(), SVC()),
                     {
                         "svc__C":[0.1,1,10],
                         "svc__gamma":[0.01,.01,.1],
                     }),


    "kNN_R":(KNeighborsRegressor,
             {
                 "n_neighbors" : (1,5,10,30,100,),
                 "weights" : ("uniform","distance",),
             }),
    
    "GB_R":(GradientBoostingRegressor,
            {
                "n_estimators" : (100,300,500,),
                "loss" : ("ls",),
                "max_depth":(3,10,),
            }),
    
    "Linear_R":(LinearRegression,
                {}),

    "Lasso_R":(Lasso,
               {
                   "alpha":(1e-4,1e-3,1e-2,)
               }),
    
}



def run_ML(ML_inputs,estimator_name = "DT_Regressor",parameters = None,by_groups = None,**kwargs):

    if estimator_name in ML_methods:
        estimator,default_parameters = ML_methods[estimator_name]
    else:
        print "\t'{}' not in estimator list:\n{}".format(estimator_name,ML_methods.keys())
        return
        

    X,y = ML_inputs.get_data("train")


    assert np.max(X) <= 1 and np.min(X) >= 0 and np.max(y) <= 1 and np.min(y) >= 0,\
        "Data is not scaled to 0-1 range"

    
    if parameters is None:
        parameters = default_parameters

    
    if by_groups is None:

        print "\nRunning ML with {}...".format(estimator_name)

        clf = estimator(**kwargs)
        
        clf = grid_search(clf,X,y,parameters)

        print "\nTrain score is {:.2f}".format(clf.score(X,y))

        f,ax = myplot(y,clf.predict(X),style = ".",shape = (1,2),figsize = (14,7),sharey = True)
        ax[0].set_title("Train samples")
        ax[0].set_ylabel("Predicted")
        
        X,y = ML_inputs.get_data("test")
        print "\nTest score is {:.2f}".format(clf.score(X,y))
        ax[1].plot(y,clf.predict(X),".")
        ax[1].set_title("Test samples")

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
    

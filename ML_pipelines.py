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
from time import time


def Lasso_Regressor(X,y,parameters = "default",**kwargs):
    
    # perform the parameter grid search to maximize cross-validation score
    clf = Lasso(**kwargs)

    if parameters == "default":
        parameters = {
            "alpha":(1e-6,1e-5,1e-4,)
        }

        clf = grid_search(clf,X,y,parameters,**kwargs)
        
    else:
        clf = grid_search(clf,X,y,parameters,**kwargs)
        
    return clf

def L_Regressor(X,y,parameters = "default",**kwargs):
    
    # perform the parameter grid search to maximize cross-validation score
    clf = LinearRegression(n_jobs = -1,**kwargs)

    if parameters == "default":
        parameters = {
        }

        clf = grid_search(clf,X,y,parameters,**kwargs)

    else:
        clf = grid_search(clf,X,y,parameters,**kwargs)
        
    return clf


def GB_Regressor(X,y,parameters = "default",**kwargs):
    
    # perform the parameter grid search to maximize cross-validation score
    clf = GradientBoostingRegressor(**kwargs)

    if parameters == "default":
        parameters = {
            "n_estimators" : (100,300,500,),
            "loss" : ("ls",),
            "max_depth":(3,10,),
            
        }

        clf = grid_search(clf,X,y,parameters,**kwargs)
        
    elif parameters is None:
        parameters = {
            "loss" : ("ls",),
        }

        clf = grid_search(clf,X,y,parameters,**kwargs)

    else:
        clf = grid_search(clf,X,y,parameters,**kwargs)
        
    return clf



def kNN_Regressor(X,y,parameters = "default",**kwargs):
    
    # perform the parameter grid search to maximize cross-validation score
    clf = KNeighborsRegressor(n_jobs = -1,**kwargs)

    if parameters == "default":
        parameters = {
            "n_neighbors" : (1,5,10,30,100,),
            "weights" : ("uniform","distance",),
        }

        clf = grid_search(clf,X,y,parameters,**kwargs)
        
    elif parameters is None:
        parameters = {
            "algorithm" : ("auto",),
        }

        clf = grid_search(clf,X,y,parameters,**kwargs)

    else:
        clf = grid_search(clf,X,y,parameters,**kwargs)
        
    return clf



def DT_Regressor(X,y,parameters = "default",**kwargs):

    
    # perform the parameter grid search to maximize cross-validation score
    clf = DecisionTreeRegressor(**kwargs)

    if parameters == "default":
        parameters = {
            "max_features" : ("sqrt","auto",),
            "min_samples_split" : (3,10,30,100,300),
        }

        clf = grid_search(clf,X,y,parameters,**kwargs)
        
    elif parameters is None:
        parameters = {
            "max_features" : ("auto",),
        }

        clf = grid_search(clf,X,y,parameters,**kwargs)

    else:
        clf = grid_search(clf,X,y,parameters,**kwargs)
        
    return clf


def DT_Classifier(X,y,parameters = "default",**kwargs):

    
    clf = DecisionTreeClassifier(**kwargs)

    if parameters == "default":
        parameters = {
            "max_features" : ("sqrt","auto",),
            "min_samples_split" : (3,10,30,100,300),
        }

        clf = grid_search(clf,X,y,parameters,**kwargs)

    else:
        clf = grid_search(clf,X,y,parameters,**kwargs)
        
    return clf 


def RF_Classifier(X,y,parameters = "default",**kwargs):

    
    # perform the parameter grid search to maximize cross-validation score
    clf = RandomForestClassifier(**kwargs)

    if parameters == "default":
        parameters = {
            "max_features" : ("sqrt","auto",),
            "min_samples_split" : (3,10,30,100,300),
            "n_estimators" : (10,20,30),            
        }

        clf = grid_search(clf,X,y,parameters,**kwargs)
    else:
        clf = grid_search(clf,X,y,parameters,**kwargs)
        
    return clf 


def RF_Regressor(X,y,parameters = None,**kwargs):

    
    # perform the parameter grid search to maximize cross-validation score
    clf = RandomForestRegressor(n_jobs = -1,**kwargs)
    if parameters is None:
        parameters = {
            "max_features" : ("sqrt","auto",),
            "min_samples_split" : (2,10,30,100,300),
            "n_estimators" : (10,20,30),
        }
    
    clf = grid_search(clf,X,y,parameters)
    
    return clf



def SV_Classifier(X,y,parameters = None,**kwargs):
    
    clf = make_pipeline(MinMaxScaler(), SVC())

    if parameters is None:
        parameters = {"svc__C":[0.1,1,10],
                      "svc__gamma":[0.01,.01,.1]}

    clf = grid_search(clf,X,y,parameters,cv = 10)

    return clf


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
        print("\n\t\tCV '{}' score (mean +/- 2*sd): {:.2f} (+/- {:.2f})".format(i,results_mean,results_std * 2))

    
    return clf


ML_methods = {
    "DT_Regressor":DT_Regressor,
    "DT_C":DT_Classifier,
    "RF_C":RF_Classifier,
    "RF_Regressor":RF_Regressor,
    "SV_Classifier":SV_Classifier,
    "kNN_R":kNN_Regressor,
    "GB_R":GB_Regressor,
    "LR":L_Regressor,
    "Lasso":Lasso_Regressor,
    
}



def run_ML(ML_inputs,estimator_name = "DT_Regressor",by_groups = None,**kwargs):

    if estimator_name in ML_methods:
        estimator = ML_methods[estimator_name]
    else:
        print "\t'{}' not in estimator list:\n{}".format(estimator_name,ML_methods.keys())
        return
        
    X,y = ML_inputs.get_data("train")

    assert np.max(X) <= 1 and np.min(X) >= 0 and np.max(y) <= 1 and np.min(y) >= 0,\
        "Data is not scaled to 0-1 range"


    if by_groups is None:
        print "\nRunning ML with {}...".format(estimator_name)
        clf = estimator(X,y,**kwargs)

        print "\nTrain score is {:.2f}".format(clf.score(X,y))
        
        X,y = ML_inputs.get_data("test")
        print "\nTest score is {:.2f}".format(clf.score(X,y))
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
    

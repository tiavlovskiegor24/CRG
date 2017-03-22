from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
#from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler,minmax_scale
#from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from time import time
#from sklearn.decomposition import PCA, KernelPCA
import numpy as np
import pandas as pd
from time import time



def RF_Classifier_pipeline(X,y,parameters = None,**kwargs):

    
    # perform the parameter grid search to maximize cross-validation score
    clf = RandomForestClassifier(**kwargs)
    if parameters is None:
        parameters = {"max_features":(5,"auto",20),"min_samples_split":(2,10,50,100)}
    
    clf = grid_search(clf,X,y,parameters)
    #best_params["n_estimators"] = 30

    # cross validate the classifier using the best parameters
    #print "\nTen-fold cross-validation scores using best parameters:"
    cv = StratifiedKFold(n_splits=10)
    scores = cross_val_score(clf.best_estimator_, X, y, cv=cv)

    # print the score of cross-validation
    #print scores
    print("\tAccuracy (mean +/- 2*sd): %0.2f (+/- %0.2f)" % (scores.mean(), \
                                           scores.std() * 2))
    print(classification_report(y, y_pred, target_names=target_names))

    return clf

def RF_Regressor_pipeline(X,y,parameters = None,**kwargs):

    
    # perform the parameter grid search to maximize cross-validation score
    clf = RandomForestRegressor(n_jobs = -1,**kwargs)
    if parameters is None:
        parameters = {
            "max_features" : ("sqrt","auto",),
            "min_samples_split" : (3,10,30,100,300),
            "n_estimators" : (10,20),
        }
    
    clf = grid_search(clf,X,y,parameters)
    #best_params["n_estimators"] = 30

    # cross validate the classifier using the best parameters
    #print "\nTen-fold cross-validation scores using best parameters:"
    cv = StratifiedKFold(n_splits=10)
    scores = cross_val_score(clf.best_estimator_, X, y, cv=cv)

    # print the score of cross-validation
    #print scores
    print("\tAccuracy (mean +/- 2*sd): %0.2f (+/- %0.2f)" % (scores.mean(), \
                                           scores.std() * 2))

    return clf



def SVM_pipeline(X,y,parameters = None,**kwargs):
    
    
    clf = make_pipeline(MinMaxScaler(), svm.SVC())
    if parameters is None:
        parameters = {"svc__C":[0.1,1,10],"svc__gamma":[0.01,.01,.1]}
    clf = grid_search(clf,X,y,parameters,cv = 10)
    print "\n\tCross-validating best parameters"
    cv = StratifiedKFold(n_splits=10)
    scores = cross_val_score(clf.best_estimator_, X, y, cv=cv)

    # print the score of cross-validation
    #print scores
    print("\tAccuracy (mean +/- 2*sd): %0.2f (+/- %0.2f)" % (scores.mean(), \
                                           scores.std() * 2))

    return clf


def grid_search(clf,X,y,parameters,**kwargs):
    
    print "\n\tPerforming parameter grid search..."
    t = time()
    clf = GridSearchCV(clf, parameters,n_jobs = -1,**kwargs)
    clf.fit(X,y)
    print "\t\tTime taken:{}".format(time()-t)
    print "\tBest parameters:",clf.best_params_
    #print "Best parameters score:",clf.best_score_

    # print the results of grid search
    '''
    cv_results = pd.DataFrame(clf.cv_results_)
    cv_results.sort_values(by="rank_test_score")
    cv_results[['params',"mean_test_score","std_test_score",\
                'rank_test_score']].sort_values(by="rank_test_score")
    print cv_results
    '''
    for i in ["train","test"]:
        scores = clf.best_index_['std_{}_score'.format(i)][clf.best_index_]
    
        print("\t'{}' accuracy (mean +/- 2*sd): %0.2f (+/- %0.2f)" % (i,scores.mean(), \
                                                             scores.std() * 2))

    
    return clf


def run_ML(ML_inputs,estimator_pipeline = RF_Regressor_pipeline,by_groups = None,**kwargs):

    X,y = ML_inputs.get_data("train")
    y = y.ravel()


    if by_groups is None:
        print "Running ML..."
        return  estimator_pipeline(X,y,**kwargs)
    else:
        
        while by_groups not in ML_inputs["sample_groups"]:
            by_group = raw_input("Select group from:\n {}\n or enter 'c' to CANCEL: "\
                              .format(ML_inputs["sample_groups"].keys()))
            if by_groups == "c":
                print "Cancelled"
                return None

        group_clfs = {}
        for name,idx in ML_inputs["sample_groups"][by_groups].iteritems():
            print "\nRunning ML for '{g}' = '{n}'".format(g = by_groups,n = name)
            X_group = X[idx]
            y_group = y[idx]
            m = y_group.shape[0]
            print "Group data set has {} samples".format(m)
            if m < 10:
                continue
            group_clfs[name] = estimator_pipeline(X_group,y_group,**kwargs)
            
        return group_clfs
    

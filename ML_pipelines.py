from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler,minmax_scale
#from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import GridSearchCV
#from sklearn import metrics
from sklearn.pipeline import make_pipeline
from time import time
#from sklearn.decomposition import PCA, KernelPCA
import numpy as np
import pandas as pd


def RF_pipeline(X,y,parameters = None,**kwargs):
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
    print("Accuracy (mean +/- 2*sd): %0.2f (+/- %0.2f)" % (scores.mean(), \
                                           scores.std() * 2))

    return clf

def SVM_pipeline(X,y,parameters = None,**kwargs):
    clf = make_pipeline(MinMaxScaler(), svm.SVC())
    if parameters is None:
        parameters = {"svc__C":[0.1,1,10],"svc__gamma":[0.01,.01,.1]}
    clf = grid_search(clf,X,y,parameters)

    cv = StratifiedKFold(n_splits=10)
    scores = cross_val_score(clf.best_estimator_, X, y, cv=cv)

    # print the score of cross-validation
    #print scores
    print("Accuracy (mean +/- 2*sd): %0.2f (+/- %0.2f)" % (scores.mean(), \
                                           scores.std() * 2))

    return clf


def grid_search(clf,X,y,parameters,cv=10):
    clf = GridSearchCV(clf, parameters,cv = cv)
    clf.fit(X,y)
    print "\nBest parameters:",clf.best_params_
    #print "Best parameters score:",clf.best_score_

    # print the results of grid search
    '''
    cv_results = pd.DataFrame(clf.cv_results_)
    cv_results.sort_values(by="rank_test_score")
    cv_results[['params',"mean_test_score","std_test_score",\
                'rank_test_score']].sort_values(by="rank_test_score")
    print cv_results
    '''
    return clf

    
    
    

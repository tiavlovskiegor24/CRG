from importlib import import_module

def get_estimator(estimator_name):

    while True:
        if estimator_name in ML_estimators:
            estimator = ML_estimators[estimator_name]
            break
        else:
            print "\t'{}' not in estimator list:\n{}"\
                .format(estimator_name,enumerate(ML_estimators.keys()))

            estimator_name = raw_input("Choose one from above (C to cancel):")
            if estimator_name == "C":
                return None,None

    return getattr(import_module(estimator["module"]),estimator["estimator"]),estimator["params"]


ML_estimators = {

    "DTree_R":{
        "estimator":"DecisionTreeRegressor",
        "module":"sklearn.tree",
        "params":{
            "max_features" : ("sqrt","auto",),
            "min_samples_split" : (3,10,30,100,300),
        },
    },

    "DTree_C":{
        "estimator":"DecisionTreeClassifier",
        "module":"sklearn.tree",
        "params":{
            "max_features" : ("sqrt","auto",),
            "min_samples_split" : (3,10,30,100,300),
        },
    },

    "RForest_C":{
        "estimator":"RandomForestClassifier",
        "module":"sklearn.ensemble",
        "params":{
                "max_features" : ("sqrt","auto",),
                "min_samples_split" : (3,10,30,100,300),
                "n_estimators" : (10,20,30),            
        },
    },

    "RForest_R":{
        "estimator":"RandomForestRegressor",
        "module":"sklearn.ensemble",
        "params":{
                "max_features" : ("sqrt","auto",),
                "min_samples_split" : (3,10,30,100,300),
                "n_estimators" : (10,20,30),            
        },
    },

    "GradBoost_R":{
        "estimator":"GradientBoostingRegressor",
        "module":"sklearn.ensemble",
        "params":{
            "n_estimators" : (100,300,500,),
            "loss" : ("ls","huber"),
            "max_depth":(1,3,10,),
        },
    },

    "SV_C":{
        "estimator":"SVC",
        "module":"sklearn.svm",
        "params":{
            "C":(0.1,1,10),
            "gamma":(0.01,.01,.1),
        },
    },


    "kNN_R":{
        "estimator":"KNeighborsRegressor",
        "module":"sklearn.neigbours",
        "params":{
            "n_neighbors" : (1,5,10,30,100,),
            "weights" : ("uniform","distance",),
        },
    },

    


    "Linear_R":{
        "estimator":"LinearRegression",
        "module":"sklearn.linear_model",
        "params":{
        },
    },


    "Lasso_R":{
        "estimator":"Lasso",
        "module":"sklearn.linear_model",
        "params":{
           "alpha":(1e-4,1e-3,1e-2,)
        },
    },
}

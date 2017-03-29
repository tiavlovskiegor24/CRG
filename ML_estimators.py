from importlib import import_module

def get_estimator(estimator_name):

    while True:
        if estimator_name in ML_estimators:
            estimator = ML_estimators[estimator_name]
            break
        else:
            print "\t'{}' not in estimator list:\n\t{}"\
                .format(estimator_name,ML_estimators.keys())

            estimator_name = raw_input("\tChoose one from above (C to cancel):")
            if estimator_name == "C":
                return None,None

    return estimator_name,getattr(import_module(estimator["module"]),estimator["estimator"]),estimator["params"]


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
                "n_estimators" : (30,),            
        },
    },


    "RForest_MC":{
        "estimator":"RandomForestClassifier",
        "module":"sklearn.ensemble",
        "params":{
                "max_features" : ("sqrt","auto",),
                "min_samples_split" : (3,10,30,100,300),
                "n_estimators" : (30,),            
        },
    },

    "RForest_R":{
        "estimator":"RandomForestRegressor",
        "module":"sklearn.ensemble",
        "params":{
                "max_features" : ("sqrt","auto",),
                "min_samples_split" : (3,10,30,100,300),
                "n_estimators" : (30,),            
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
            "C":(0.2,0.3,0.4),
            "gamma":(0.008,0.01,0.012,),
        },
    },


    "kNN_R":{
        "estimator":"KNeighborsRegressor",
        "module":"sklearn.neighbors",
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
           "alpha":(3e-5,1e-4,3e-4,1e-3,)
        },
    },

    "Log_C":{
        "estimator":"LogisticRegression",
        "module":"sklearn.linear_model",
        "params":{
            "C":(1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1e0,3e0,1e1,3e1,1e2,3e2)
        }
    },

    "OneVsRest_MC":{
        "estimator":"sklearn.multiclass",
        "module":"OneVsRestClassifier",
        "params":{
            "base_estimator":"log_C",
        }
    },
}

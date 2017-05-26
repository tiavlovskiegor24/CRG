from importlib import import_module
import numpy as np

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

    return estimator_name,getattr(import_module(estimator["module"]),estimator["estimator"]),estimator["param_grid"]


ML_estimators = {

    "DTree_R":{
        "estimator":"DecisionTreeRegressor",
        "module":"sklearn.tree",
        "param_grid":{
            "max_features" : ("sqrt","auto",),
            "min_samples_split" : (3,10,30,100,300),
        },
    },

    "DTree_C":{
        "estimator":"DecisionTreeClassifier",
        "module":"sklearn.tree",
        "param_grid":{
            "max_features" : ("sqrt","auto",),
            "min_samples_split" : (3,10,30,100,300),
        },
    },

    "RForest_C":{
        "estimator":"RandomForestClassifier",
        "module":"sklearn.ensemble",
        "param_grid":{
                "max_features" : ("sqrt","auto",),
                "min_samples_split" : (3,10,30,100,300),
                "n_estimators" : (30,),            
        },
    },


    "RForest_MC":{
        "estimator":"RandomForestClassifier",
        "module":"sklearn.ensemble",
        "param_grid":{
                "max_features" : ("sqrt","auto",),
                "min_samples_split" : (3,10,30,100,300),
                "n_estimators" : (30,),            
        },
    },

    "RForest_R":{
        "estimator":"RandomForestRegressor",
        "module":"sklearn.ensemble",
        "param_grid":{
                "max_features" : ("sqrt","auto",),
                "min_samples_split" : (3,10,30,100,300),
                "n_estimators" : (30,),            
        },
    },

    "GradBoost_R":{
        "estimator":"GradientBoostingRegressor",
        "module":"sklearn.ensemble",
        "param_grid":{
            "n_estimators" : (100,300,500,),
            "loss" : ("ls","huber"),
            "max_depth":(1,3,10,),
        },
    },

    "SV_C":{
        "estimator":"SVC",
        "module":"sklearn.svm",
        "param_grid":{
            "C":[1*(3**j)*(10**i) for i in range(-3,2) for j in [0,1]],
            "gamma":[1*(3**j)*(10**i) for i in range(-3,2) for j in [0,1]],
        },
    },

    "SV_R":{
        "estimator":"SVR",
        "module":"sklearn.svm",
        "param_grid":{
            "C":[1*(3**j)*(10**i) for i in range(-3,2) for j in [0,1]],
            "epsilon":[1*(3**j)*(10**i) for i in range(-3,2) for j in [0,1]],
        },
    },
    

    "kNN_R":{
        "estimator":"KNeighborsRegressor",
        "module":"sklearn.neighbors",
        "param_grid":{
            "n_neighbors" : (1,5,10,30,100,),
            "weights" : ("uniform","distance",),
        },
    },

    
    "Linear_R":{
        "estimator":"LinearRegression",
        "module":"sklearn.linear_model",
        "param_grid":{
        },
    },

    "Ridge_R":{
        "estimator":"Ridge",
        "module":"sklearn.linear_model",
        "param_grid":{
           "alpha":(1e-4,3e-4,1e-3,3e-3,1e-2)
        },
    },


    "Lasso_R":{
        "estimator":"Lasso",
        "module":"sklearn.linear_model",
        "param_grid":{
           "alpha":(1e-4,3e-4,1e-3,3e-3,1e-2)
        },
    },

    "ENet_R":{
        "estimator":"ElasticNet",
        "module":"sklearn.linear_model",
        "param_grid":{
            "alpha":(1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3),
            "l1_ratio":np.linspace(0.0,1.0,num = 10)
        },
    },

    
    "Log_C":{
        "estimator":"LogisticRegression",
        "module":"sklearn.linear_model",
        "param_grid":{
            "C":(1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1e0,3e0,1e1,3e1,1e2,3e2)
        }
    },

    "OneVsRest_MC":{
        "estimator":"sklearn.multiclass",
        "module":"OneVsRestClassifier",
        "param_grid":{
            "base_estimator":"log_C",
        }
    },
}

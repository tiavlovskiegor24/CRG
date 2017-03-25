import ML_inputs
import ML_process

if __name__ == "__main__":
    import control_file as cf
    import feature_types as f_types
    ml_inputs = ML_inputs.get_ML_inputs(cf = cf,f_types = f_types)
    clf = ML_process.run_ML(ml_inputs,estimator_name = "Linear_R")
    

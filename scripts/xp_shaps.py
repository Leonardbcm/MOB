import os, itertools, pandas, numpy as np, time, warnings

from src.models.spliter import MySpliter
from src.models.torch_wrapper import OBNWrapper
from src.models.torch_models.weight_initializers import *
import src.models.parallel_scikit as ps

####### MAIN PARAMS
XP_PARAMS = {
    "START": 0,
    "STOP" : 365,
    "STEP" : 30,
    "N_SHAP" : 1000,    
    "N_VAL": 365,
    "BATCH" : 80,
    "tboard" : "BETA_RESULTS",
}

CHECK_PARAMS = {
    "START": 0,
    "STOP" : 30,
    "STEP" : 30,
    "N_VAL": 365, 
    "BATCH" : 80,   
    "N_SHAP" : 2,    
    "tboard" : "BETA_RESULTS",
}

PARAMS = XP_PARAMS

####### Configs to test
#countries = ["FR", "DE", "BE", "NL"]
#datasets = ["Lyon", "Munich", "Bruges", "Lahaye"]
countries = ["BE"]
datasets = ["Bruges"]
IDs = [4, 6, 8]
OB_sizes = [20]
seeds = [0]

#### LOOP
n = len(IDs)
nObs = len(OB_sizes)
for i, ID in enumerate(IDs):
    for k, OBs in enumerate(OB_sizes):
        for j, (country, dataset) in enumerate(zip(countries, datasets)):
            for seed in seeds:
                spliter = MySpliter(PARAMS["N_VAL"], shuffle=False)        
                model_wrapper = OBNWrapper(
                    "XP",dataset,spliter=spliter, country=country,
                    skip_connection=True,
                    use_order_books=False, order_book_size=OBs, IDn=ID,
                    tboard=PARAMS["tboard"])
                X, Y = model_wrapper.load_train_dataset()
                (Xtr, Ytr), (Xv, Yv) = model_wrapper.spliter(X, Y)                

                # LOAD DATA AND RESTORE REGRESSOR STEP
                regr, version = model_wrapper.load(X, Y)
                
                # PREPARE THE EXPLAINER
                explainer = model_wrapper.prepare_explainer(regr, Xtr)
                warnings.filterwarnings("ignore")
                
                shaps = np.zeros(
                    (model_wrapper.N_OUTPUT, PARAMS["STOP"] - PARAMS["START"],
                     Xv.shape[1]), dtype=np.float16)
                
                # COMPUTE SHAP VALUES BY BATCH
                for i in range(PARAMS["START"], PARAMS["STOP"], PARAMS["STEP"]):
                    shap_values = explainer.shap_values(
                        X=Xv[i:i+PARAMS["STEP"]],nsamples=PARAMS["N_SHAP"],
                        silent=True, l1_reg="num_features(25)", gc_collect=True)
    
                    np.save(model_wrapper.tr_shap_path(version, i), shap_values)
                    for l in range(model_wrapper.N_OUTPUT):
                        shaps[l, i:i+PARAMS["STEP"]] = shap_values[l]

                # SAVE ALL SHAP VALUES
                np.save(model_wrapper.test_recalibrated_shape_path(version), shaps)



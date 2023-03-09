from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor

from src.models.spliter import MySplitter
from src.models.model_wrapper import *
from src.models.obn.obn import OrderBookNetwork

class OBNWrapper(ModelWrapper):
    def __init__(self, prefix, dataset_name, spliter=None,
                 predict_two_days=False, replace_ATC="",
                 known_countries=["CH", "GB"], countries_to_predict="all"):
        ModelWrapper.__init__(self, prefix, dataset_name,
                              spliter=spliter, predict_two_days=predict_two_days,
                              known_countries=known_countries,
                              replace_ATC=replace_ATC,
                              countries_to_predict=countries_to_predict)
        if spliter is None: spliter = MySplitter(0.25)        
        self.spliter = spliter
        self.external_spliter = None        
        self.validation_mode = "internal"
        
    def params(self):
        return {
            # Network Architecture
            "N_OUTPUT" : len(self.label),            
            "NN1" : (900, 888),
            "OBs" : 50,
            "OBN" : (100, ),            
            "k" : 30,
            "niter" : 20,
            "batch_solve" : True, 
            "batch_norm" : True,
            "pmin" : -500,
            "pmax" : 3000,
            "step" : 0.01,
            "mV" : 0.1,
            "check_data" : False,
            "clip" : False,
            "scale" : False,
            "store_OBhat" : False,

            # Log params
            "store_losses" : False,
            "tensorboard" : "",
            
            # Scaling Parameters
            "scaler" : "BCM",
            "transformer" : "Standard",

            # Training Params            
            "spliter" : self.spliter, 
            "n_epochs" : 100000,
            "batch_size" : 30,
            "early_stopping" : "sliding_average",
            "early_stopping_alpha" : 20,
            "early_stopping_patience" : 20,
            "shuffle_train" : True,

            # Optimizer Params
            "criterion" : "HuberLoss",
        }

    def make(self, ptemp):
        scaler, transformer, ptemp_ = self.prepare_for_make(ptemp)
        ptemp_["transformer"] = transformer
        
        # OBN need to access the scalers for scaling OBN
        model = OrderBookNetwork("test", ptemp_)
        pipe = make_pipeline(scaler, model)
        return pipe

    def get_search_space(self, country, version=None,  n=None, fast=False,
                         stop_after=-1):
        return OBN_space(n, country, fast=fast, stop_after=stop_after)

    def predict_order_books(self, regr, X):
        model = regr.steps[1][1]
        scaler = regr.steps[0][1]
        
        OBhat = model.predict_order_books(scaler.transform(X))
        return OBhat    
    
    def string(self):
        return "OBN"

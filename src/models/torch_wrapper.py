from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor

from src.models.spliter import MySpliter
from src.models.model_wrapper import *
from src.models.torch_models.obn import OrderBookNetwork


class TorchWrapper(ModelWrapper):
    """
    Base wrapper for all torch-based models
    """
    def __init__(self, prefix, dataset_name, spliter=None, country=""):
        ModelWrapper.__init__(
            self, prefix, dataset_name, spliter=spliter, country=country)


class OBNWrapper(TorchWrapper):
    """
    Wrapper for all predict order books then optimize models
    """    
    def __init__(self, prefix, dataset_name, spliter=None, country=""):
        TorchWrapper.__init__(
            self, prefix, dataset_name, spliter=spliter, country=country)
        if spliter is None: spliter = MySpliter(0.25)        
        self.spliter = spliter
        self.external_spliter = None        
        self.validation_mode = "internal"
        
    def params(self):
        return {
            # Network Architecture
            "N_OUTPUT" : len(self.label),            
            "NN1" : (888, ),
            "OBs" : 100,
            "OBN" : (37, ),            
            "k" : 100,
            "niter" : 30,
            "batch_solve" : True, 
            "batch_norm" : True,
            "pmin" : -500,
            "pmax" : 3000,
            "step" : 0.01,
            "mV" : 0.1,
            "check_data" : False,

            # Log params
            "store_OBhat" : False,            
            "store_losses" : False,
            "tensorboard" : "",
            
            # Scaling Parameters
            "scaler" : "BCM",
            "transformer" : "Standard",
            "OB_weight_initializers" : None,
            "scale" : False,            

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
            "n_cpus" : -1,
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

    def refit(self, regr, X, y, epochs=1):
        model = regr.steps[1][1]
        scaler = regr.steps[0][1]
        
        model.refit(scaler.transform(X), y, epochs=epochs)                
    
    def string(self):
        return "OBN"
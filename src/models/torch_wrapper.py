from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor

from src.models.spliter import MySpliter
from src.models.model_wrapper import *
from src.models.torch_models.obn import OrderBookNetwork

from src.models.samplers.combined_sampler import combined_sampler, list_combined_sampler
from src.models.samplers.structure_sampler import structure_sampler
from src.models.samplers.obn_structure_sampler import obn_structure_sampler
from src.models.samplers.regularization_sampler import regularization_sampler
from src.models.samplers.discrete_log_uniform import discrete_loguniform
from src.models.samplers.weight_initializer_samplers import wi_sampler, bi_sampler, wibi_sampler


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
    def __init__(self, prefix, dataset_name, pmin=-500, pmax=3000,
                 spliter=None, country=""):
        TorchWrapper.__init__(
            self, prefix, dataset_name, spliter=spliter, country=country)
        if spliter is None: spliter = MySpliter(0.25)

        self.pmin = pmin
        self.pmax = pmax        
        self.spliter = spliter
        self.external_spliter = None        
        self.validation_mode = "internal"
        
    def params(self):
        return {
            # Network Architecture
            "N_OUTPUT" : len(self.label),
            "NN1" : (888, ),
            "OBN" : (37, ),
            "OBs" : 100,
            "k" : 100,
            "niter" : 30,
            "batch_solve" : True, 
            "batch_norm" : True,
            "pmin" : self.pmin,
            "pmax" : self.pmax,
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
            "weight_initializers" : [],
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

    def get_search_space(self, n=None, fast=False, stop_after=-1):
        space = {
            "structure" : combined_sampler(
                [
                    obn_structure_sampler(n, 1, 0,  25, 60),
                    obn_structure_sampler(n, 1, 1,  25, 60),
                    obn_structure_sampler(n, 1, 2,  25, 60),

                    obn_structure_sampler(n, 2, 0,  25, 60),
                    obn_structure_sampler(n, 2, 1,  25, 60),
                    obn_structure_sampler(n, 2, 2,  25, 60),                    
                ],
                weights = [2, 2, 2, 1, 1, 1]),
            "OBs" : discrete_loguniform(50, 500),
            "weight_initializer" : list_combined_sampler(
                [
                    wibi_sampler(self.pmin, self.pmax),
                    wi_sampler(),
                    bi_sampler(self.pmin, self.pmax),
                    []
                ],
                weights = [4, 1, 2, 3]
            ),
        }
        if fast:
            space["n_epochs"] = [2]
            space["early_stopping"] = [""]               
        return space

    def map_dict(self):
        orig = NeuralNetWrapper.map_dict(self)
        orig.update({"structure" :
                     {
                         "OBN" : (mu.neurons_per_layer_to_string,
                                  mu.neurons_per_layer_from_string),
                         "NN1" : (mu.neurons_per_layer_to_string,
                                  mu.neurons_per_layer_from_string),
                     },
                     "weight_initializer" : (mu.weight_initializer_to_string,
                                             mu.weight_initializer_from_string) 
        })
        return orig       

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

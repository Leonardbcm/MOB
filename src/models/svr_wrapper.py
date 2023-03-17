from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
import scipy.stats as stats

from src.models.model_wrapper import *

class SVRWrapper(ModelWrapper):
    """
    Single-output svr
    """
    def __init__(self, prefix, dataset_name, country="", spliter=None):
        ModelWrapper.__init__(
            self, prefix, dataset_name, spliter=spliter, country=country)
        
    def params(self):
        return {"kernel":"rbf",
                "tol":0.0001,
                "C":1.0,
                "gamma":"scale",
                "epsilon":0.01,
                "shrinking":True,
                "max_iter":500000,
                "cache_size":200,
                "scaler":"Standard",
                "transformer":"Standard"}

    def make(self, ptemp):
        scaler, transformer, ptemp_ = self.prepare_for_make(ptemp)
        
        model = svm.SVR(**ptemp_)
        pipe = make_pipeline(scaler, model)
        regr = TransformedTargetRegressor(pipe, transformer=transformer)
        
        return regr  

    def get_search_space(self, n=None, fast=False, stop_after=None):
            space = {
                "C" : stats.loguniform(10**-9, 10**9),
                "gamma" : stats.loguniform(10**-9, 10**9),
                "scaler" : ["BCM", "Standard", "Median", "SinMedian"],
                "transformer" : ["BCM", "Standard", "Median", "SinMedian"]
            }
            
            if fast:
                space["max_iter"] = [2]

            return space
            
 
    def string(self):
        return "SVR"
    
    
class MultiSVRWrapper(SVRWrapper):
    """
    Multivariuate output svr using MultiOutputRegressor
    """    
    def __init__(self, prefix, dataset_name, spliter=None, country=""):
        SVRWrapper.__init__(
            self, prefix, dataset_name, spliter=spliter, country=country)

    def make(self, ptemp):
        scaler, transformer, ptemp_ = self.prepare_for_make(ptemp)

        model = MultiOutputRegressor(svm.SVR(**ptemp_), n_jobs=-1)
        pipe = make_pipeline(scaler, model)
        regr = TransformedTargetRegressor(pipe, transformer=transformer)        
        return regr    

    def string(self):
        return "SVRMulti"
    
class ChainSVRWrapper(SVRWrapper):
    """
    Multivariuate output svr using ChainRegressor (forecast #1 is used to forecast
    #2, and so on.
    """     
    def __init__(self, prefix, dataset_name, spliter=None, country=""):
        SVRWrapper.__init__(
            self, prefix, dataset_name, spliter=spliter, country=country)

    def make(self, ptemp):
        scaler, transformer, ptemp_ = self.prepare_for_make(ptemp)
        
        model = RegressorChain(svm.SVR(**ptemp_))
        pipe = make_pipeline(scaler, model)
        regr = TransformedTargetRegressor(pipe, transformer=transformer)

        return regr    
    
    def string(self):
        return "SVRChain"

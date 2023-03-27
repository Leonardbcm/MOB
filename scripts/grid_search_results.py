from src.models.svr_wrapper import ChainSVRWrapper, MultiSVRWrapper
from src.models.torch_wrapper import OBNWrapper
from src.models.keras_wrapper import DNNWrapper, CNNWrapper
from src.models.grid_search import *
from src.models.spliter import MySpliter

models = [ChainSVRWrapper, MultiSVRWrapper, DNNWrapper, CNNWrapper, OBNWrapper]
spliter = MySpliter(365)
for model in models:
    model_wrapper = create_mw("", "Lyon", model, "TSCHORA", EPF="", spliter=spliter)
    results = model_wrapper.load_results()
    print(model.string(model))
    model_wrapper.best_params(results)    

import os, itertools, pandas, numpy as np, time

from src.models.spliter import MySpliter
from src.models.torch_wrapper import OBNWrapper
from src.models.torch_models.weight_initializers import *
import src.models.parallel_scikit as ps

"""
Main XP file. For each listed configuration evaluate the base params on all 
datasets.
"""
####### MAIN PARAMS
N_EPOCHS = 1000
N_SAMPLES = 1444
N_VAL = 365
BATCH = 80

####### Results container
results = pandas.DataFrame(
    columns=[
        "country", "ID",
        "val_price_mae", "val_price_smape", "val_price_ACC", "val_OB_smape",
        "val_OB_ACC",    
        "training_time"])
####### configurations
countries = ["FR", "DE", "BE", "NL"]
datasets = ["Lyon", "Munich", "Bruges", "Lahaye"]
IDs = [1, 2, 3, 4, 5, 6, 7]
######## For storing results
n = 7
for i, ID in enumerate(IDs):
    for j, (country, dataset) in enumerate(zip(countries, datasets)):
        spliter = MySpliter(N_VAL, shuffle=False)        
        model_wrapper = OBNWrapper(
            "XP", dataset, spliter=spliter, country=country, skip_connection=True,
            use_order_books=False, order_book_size=20, IDn=ID, tboard="")
        
        print(model_wrapper.logs_path)

        # Load train dataset
        X, Y = model_wrapper.load_train_dataset()
        X = X[:N_SAMPLES, :]
        Y = Y[:N_SAMPLES, :]
        (_, _), (Xv, Yv) = model_wrapper.spliter(X, Y)

        # Create the model with the default params
        default_params = model_wrapper.params()        
        default_params["early_stopping"] = None
        default_params["n_epochs"] = N_EPOCHS
        default_params["batch_size"] = BATCH        
        default_params["OB_plot"] = False
        default_params["profile"] = False
        regr = model_wrapper.make(default_params)

        # Fit the model
        start = time.time()
        regr.fit(X, Y)
        stop = time.time()

        # Predict validation data and compute errors
        yvpred = model_wrapper.predict_val(regr, Xv)
        
        if not model_wrapper.predict_order_books:
            price_mae = model_wrapper.price_mae(Yv, yvpred)
            price_smape = model_wrapper.price_smape(Yv, yvpred)        
            price_acc = model_wrapper.price_ACC(Yv, yvpred)
        else:
            price_mae = np.nan
            price_smape = np.nan
            price_acc = np.nan

        if model_wrapper.gamma > 0:
            OB_smape = model_wrapper.OB_smape(Yv, yvpred)        
            OB_acc = model_wrapper.OB_ACC(Yv, yvpred)
        else:
            OB_smape = np.nan
            OB_acc = np.nan

        # Save validation data
        train_dates, validation_dates = spliter(model_wrapper.train_dates)
        pandas.DataFrame(yvpred, index=validation_dates).to_csv(
            model_wrapper.validation_prediction_path())

        # Also save in the log folder
        pandas.DataFrame(yvpred, index=validation_dates).to_csv(
            os.path.join(model_wrapper.logs_path, model_wrapper.latest_version,
                         "validation_predictions.csv"))        

        res = pandas.DataFrame({
            "country" : country,
            "ID" : ID,                        
            "val_price_mae" : price_mae,
            "val_price_smape" : price_smape,            
            "val_price_ACC" : price_acc,
            "val_OB_smape" : OB_smape,                        
            "val_OB_ACC" : OB_acc,
            "training_time" : stop - start,            
        }, index = [n * j + i])
        results = pandas.concat([results, res], ignore_index=True)

results.to_csv()

import itertools, pandas, numpy as np, time

from src.models.spliter import MySpliter
from src.models.torch_wrapper import OBNWrapper
from src.models.torch_models.weight_initializers import *
import src.models.parallel_scikit as ps

"""
Main XP file. For each listed configuration evaluate the base params on all 
datasets.
"""

####### Results container
results = pandas.DataFrame(
    columns=["country", "skip_connection", "use_order_book",
             "order_book_size", "separate_optim",
             "val_mae", "val_smape", "val_ACC",
             "test_mae", "test_smape", "test_ACC",
             "training_time"])
####### configurations
#skip_connections = [True, False]
skip_connections = [True]

#use_order_books = [True, False]
use_order_books = [False]

#separate_optims = [True, False]
separate_optims = [True]

#order_book_sizes = [20, 50, 100, 250]
order_book_sizes = [20, 50]

#countries = ["FR", "DE", "BE", "NL"]
countries = ["BE"]

#datasets = ["Lyon", "Munich", "Bruges", "Lahaye"]
datasets = ["Bruges"]

combinations = list(itertools.product(
    skip_connections, use_order_books, separate_optims, order_book_sizes))

combinations = [(True, False, True, 20),
                (True, False, True, 50)]
####### Default params
params = {}
params["n_epochs"] = 1000
######## For storing results
spliter = MySpliter(365, shuffle=False)
n = len(skip_connections) * len(use_order_books) * len(separate_optims) * len(order_book_sizes)
for i, (skip_connection, use_order_book,  separate_optim, order_book_size) in enumerate(combinations):
    for j, (country, dataset) in enumerate(zip(countries, datasets)):
        model_wrapper = OBNWrapper(
            "TEST", dataset, spliter=spliter, country=country,
            skip_connection=skip_connection, use_order_books=use_order_book,
            order_book_size=order_book_size, separate_optim=separate_optim)
        print(model_wrapper.logs_path)

        # Load train dataset
        X, Y = model_wrapper.load_train_dataset()
        (_, _), (Xv, Yv) = model_wrapper.spliter(X, Y)

        # Create the model with the default params
        default_params = model_wrapper.params()
        default_params.update(params)
        regr = model_wrapper.make(model_wrapper._params(default_params))

        # Fit the model
        start = time.time()
        regr.fit(X, Y)
        stop = time.time()

        # Predict validation data and compute errors
        yvpred = model_wrapper.predict_val(regr, Xv)
        v_mae = model_wrapper.mae(Yv, yvpred)
        v_smape = model_wrapper.smape(Yv, yvpred)        
        v_acc = model_wrapper.ACC(Yv, yvpred)

        # Save validation data
        train_dates, validation_dates = spliter(model_wrapper.train_dates)
        pandas.DataFrame(yvpred, index=validation_dates).to_csv(
            model_wrapper.validation_prediction_path())

        # Load Test dataset
        Xt, Yt = model_wrapper.load_test_dataset()

        # Predict test data and compute errors
        ytpred = model_wrapper.predict_test(regr, Xt)
        t_mae = model_wrapper.mae(Yt, ytpred)
        t_smape = model_wrapper.smape(Yt, ytpred)        
        t_acc = model_wrapper.ACC(Yt, ytpred)

        # Save test forecasts
        test_dates =  model_wrapper.test_dates
        pandas.DataFrame(ytpred, index=test_dates).to_csv(
            model_wrapper.test_prediction_path())

        res = pandas.DataFrame({
            "country" : country,
            "skip_connection": skip_connection,
            "use_order_book": use_order_book,
            "order_book_size": order_book_size,
            "separate_optim": separate_optim,
            "val_mae" : v_mae,
            "val_smape" : v_smape,            
            "val_ACC" : v_acc,
            "test_mae" : t_mae,
            "test_smape" : t_smape,                        
            "test_ACC" : t_acc,
            "training_time" : stop - start,            
        }, index = [n * j + i])
        results = pandas.concat([results, res], ignore_index=True)

########################################### 2ND ROUND
####### configurations
#skip_connections = [True, False]
skip_connections = [True]

#use_order_books = [True, False]
use_order_books = [False]

#separate_optims = [True, False]
separate_optims = [True]

#order_book_sizes = [20, 50, 100, 250]
order_book_sizes = [20, 50]

#countries = ["FR", "DE", "BE", "NL"]
countries = ["FR", "DE"]

#datasets = ["Lyon", "Munich", "Bruges", "Lahaye"]
datasets = ["Lyon", "Munich"]

combinations = list(itertools.product(
    skip_connections, use_order_books, separate_optims, order_book_sizes))

combinations = [(True, False, True, 20)]
####### Default params
params = {}
params["n_epochs"] = 1000
######## For storing results
spliter = MySpliter(365, shuffle=False)
n = len(skip_connections) * len(use_order_books) * len(separate_optims) * len(order_book_sizes)
for i, (skip_connection, use_order_book,  separate_optim, order_book_size) in enumerate(combinations):
    for j, (country, dataset) in enumerate(zip(countries, datasets)):
        model_wrapper = OBNWrapper(
            "TEST", dataset, spliter=spliter, country=country,
            skip_connection=skip_connection, use_order_books=use_order_book,
            order_book_size=order_book_size, separate_optim=separate_optim)
        print(model_wrapper.logs_path)

        # Load train dataset
        X, Y = model_wrapper.load_train_dataset()
        (_, _), (Xv, Yv) = model_wrapper.spliter(X, Y)

        # Create the model with the default params
        default_params = model_wrapper.params()
        default_params.update(params)
        regr = model_wrapper.make(model_wrapper._params(default_params))

        # Fit the model
        start = time.time()
        regr.fit(X, Y)
        stop = time.time()

        # Predict validation data and compute errors
        yvpred = model_wrapper.predict_val(regr, Xv)
        v_mae = model_wrapper.mae(Yv, yvpred)
        v_smape = model_wrapper.smape(Yv, yvpred)        
        v_acc = model_wrapper.ACC(Yv, yvpred)

        # Save validation data
        train_dates, validation_dates = spliter(model_wrapper.train_dates)
        pandas.DataFrame(yvpred, index=validation_dates).to_csv(
            model_wrapper.validation_prediction_path())

        # Load Test dataset
        Xt, Yt = model_wrapper.load_test_dataset()

        # Predict test data and compute errors
        ytpred = model_wrapper.predict_test(regr, Xt)
        t_mae = model_wrapper.mae(Yt, ytpred)
        t_smape = model_wrapper.smape(Yt, ytpred)        
        t_acc = model_wrapper.ACC(Yt, ytpred)

        # Save test forecasts
        test_dates =  model_wrapper.test_dates
        pandas.DataFrame(ytpred, index=test_dates).to_csv(
            model_wrapper.test_prediction_path())

        res = pandas.DataFrame({
            "country" : country,
            "skip_connection": skip_connection,
            "use_order_book": use_order_book,
            "order_book_size": order_book_size,
            "separate_optim": separate_optim,
            "val_mae" : v_mae,
            "val_smape" : v_smape,            
            "val_ACC" : v_acc,
            "test_mae" : t_mae,
            "test_smape" : t_smape,                        
            "test_ACC" : t_acc,
            "training_time" : stop - start,            
        }, index = [n * j + i])
        results = pandas.concat([results, res], ignore_index=True)

        

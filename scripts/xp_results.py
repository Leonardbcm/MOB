%load aimport
import itertools, pandas, numpy as np, datetime, matplotlib.pyplot as plt, time
import matplotlib

from src.models.spliter import MySpliter
from src.models.torch_wrapper import OBNWrapper
from src.models.torch_models.weight_initializers import *
import src.models.parallel_scikit as ps
from src.analysis.utils import *
from src.analysis.evaluate import DM
from src.analysis.xp_results_utils import *

"""
XP results and analysis file
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
        "val_OB_ACC"])
####### configurations
countries = ["FR", "DE", "BE", "NL"]
datasets = ["Lyon", "Munich", "Bruges", "Lahaye"]
IDs = [1, 2, 3, 4, 5, 6, 7]

######## For retrieving results
nc = len(countries)
nh = 24
n = 7
OBs = 20

real_prices = np.zeros((nc, n, N_VAL, nh))
real_OB =  np.zeros((nc, n, N_VAL, 72*OBs))

predicted_prices = np.zeros((nc, n, N_VAL, nh))
predicted_OB = np.zeros((nc, n, N_VAL, 72*OBs))

for j, (country, dataset) in enumerate(zip(countries, datasets)):
    for i, ID in enumerate(IDs):

        ###### Create Model wrapper
        spliter = MySpliter(N_VAL, shuffle=False)        
        model_wrapper = OBNWrapper(
            "RESULTS", dataset, spliter=spliter, country=country,
            skip_connection=True, use_order_books=False, order_book_size=OBs,
            IDn=ID, tboard="RESULTS")
        version = model_wrapper.highest_version
        stopped_epoch = model_wrapper.get_stopped_epoch(version)

        ###### Load DATA        
        X, Y = model_wrapper.load_train_dataset()
        X = X[:N_SAMPLES, :]
        Y = Y[:N_SAMPLES, :]
        (_, _), (Xv, Yv) = model_wrapper.spliter(X, Y)

        ###### Compute metrics
        yvpred = model_wrapper.get_predictions(version).values
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

        ###### Store results            
        res = pandas.DataFrame({
            "country" : country,
            "ID" : ID,                        
            "val_price_mae" : price_mae,
            "val_price_smape" : price_smape,            
            "val_price_ACC" : price_acc,
            "val_OB_smape" : OB_smape,                        
            "val_OB_ACC" : OB_acc,
        }, index = [n * j + i])
        results = pandas.concat([results, res], ignore_index=True)

        ###### Store predictions and labels
        if not model_wrapper.predict_order_books:
            predicted_prices[j, i] = yvpred[:, model_wrapper.y_indices]
            real_prices[j, i] = Yv[:, model_wrapper.y_indices]
            
        if model_wrapper.gamma > 0:
            predicted_OB[j, i] = yvpred[:, model_wrapper.yOB_indices]
            real_OB[j, i] = Yv[:, model_wrapper.yOB_indices]            
            

results = results.sort_values(["country", "ID"]).loc[:, ["country", "ID", "val_OB_smape", "val_OB_ACC", "val_price_mae", "val_price_smape", "val_price_ACC"]]
print(df_to_latex(results.set_index(["country", "ID"]), roundings=[2, 2, 3, 2, 2, 3], hlines=False))


####### Compute DM tests
n_prices = 6
n_OBs = 4

prices_pvalues = np.zeros((nc, n_prices, n_prices))
OB_pvalues = np.zeros((nc, n_OBs, n_OBs))

for i, (country, dataset) in enumerate(zip(countries, datasets)):
    current_prices_1 = -1
    current_OB_1 = -1

    IDs_prices = []
    IDs_OBs = []    
    for j, ID1 in enumerate(IDs):
        model_wrapper_1 = OBNWrapper(
            "RESULTS", dataset, country=country, IDn=ID1, tboard="RESULTS",
            skip_connection=True, use_order_books=False, order_book_size=OBs)

        if model_wrapper_1.gamma > 0:
            current_OB_1 += 1
            IDs_OBs.append(ID1)            
        if not model_wrapper_1.predict_order_books:
            current_prices_1 += 1
            IDs_prices.append(ID1)
            
        current_prices_2 = 0
        current_OB_2 = 0
        
        for k, ID2 in enumerate(IDs):
            model_wrapper_2 = OBNWrapper(
                "RESULTS", dataset, country=country, IDn=ID2, tboard="RESULTS",
                skip_connection=True, use_order_books=False, order_book_size=OBs)

            # Compute the DM test on the Orderbooks
            if (model_wrapper_1.gamma > 0) and (model_wrapper_2.gamma > 0):
                Y = real_OB[i, current_OB_1]
                Yhat1 = predicted_OB[i, j]
                Yhat2 = predicted_OB[i, k]                
                if ID1 == ID2:
                    OB_pvalue = np.nan
                else:
                    OB_pvalue = DM(Y, Yhat1, Yhat2, norm="smape")
                    
                OB_pvalues[i, current_OB_1, current_OB_2] = OB_pvalue
                current_OB_2 += 1                

            # Compute the DM test on the Prices
            if (not model_wrapper_1.predict_order_books) and (not model_wrapper_2.predict_order_books):
                Y = real_prices[i, current_prices_1]
                Yhat1 = predicted_prices[i, j]
                Yhat2 = predicted_prices[i, k]                
                if ID1 == ID2:
                    prices_pvalue = np.nan
                else:
                    prices_pvalue = DM(Y, Yhat1, Yhat2, norm="mae")
                    
                prices_pvalues[i, current_prices_1,current_prices_2] = prices_pvalue
                current_prices_2 += 1

with matplotlib.rc_context({ "text.usetex" : True,
                             "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                             "font.family" : ""}):
    plt.close("all")    
    plot_DM_tests(
        prices_pvalues, countries=countries, IDs=IDs_prices, label="prices")
    plt.show()

with matplotlib.rc_context({ "text.usetex" : True,
                             "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                             "font.family" : ""}):
    plt.close("all")    
    plot_DM_tests(
    OB_pvalues, countries=countries, IDs=IDs_OBs, label="Order Books")
    plt.show()
    

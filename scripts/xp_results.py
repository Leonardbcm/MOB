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

####### configurations
#countries = ["FR", "DE", "BE", "NL"]
countries = ["FR"]
#datasets = ["Lyon", "Munich", "Bruges", "Lahaye"]
datasets = ["Lyon"]
IDs = [1, 2, 3, 4, 5, 6, 7]

######## For retrieving results, OBs = 20
predicted_prices_20, real_prices_20, predicted_OB_20, real_OB_20, results_20 = retrieve_results(IDs, countries, datasets, 20, N_VAL, N_SAMPLES, nh = 24)

######## For retrieving results, OBs = 50
predicted_prices_50, real_prices_50, predicted_OB_50, real_OB_50, results_50 = retrieve_results(IDs, countries, datasets, 50, N_VAL, N_SAMPLES, nh = 24)

key_order = ["country", "ID", "val_OB_smape", "val_OB_ACC", "val_price_mae",
             "val_price_smape", "val_price_ACC"]
results = results.sort_values(["country", "ID"]).loc[:, key_order]
print(df_to_latex(
    results.set_index(["country", "ID"]),
    roundings=[2, 2, 3, 2, 2, 3], hlines=False))

####### Compute DM tests
prices_pvalues = np.ones((nc, n, n)) * np.nan
OB_pvalues = np.ones((nc, n, n))  * np.nan

for i, (country, dataset) in enumerate(zip(countries, datasets)):
    for j, ID1 in enumerate(IDs):
        model_wrapper_1 = OBNWrapper(
            "RESULTS", dataset, country=country, IDn=ID1, tboard="RESULTS",
            skip_connection=True, use_order_books=False, order_book_size=OBs)
        
        for k, ID2 in enumerate(IDs):
            model_wrapper_2 = OBNWrapper(
                "RESULTS", dataset, country=country, IDn=ID2, tboard="RESULTS",
                skip_connection=True, use_order_books=False, order_book_size=OBs)

            # Compute the DM test on the Orderbooks
            if (model_wrapper_1.gamma > 0) and (model_wrapper_2.gamma > 0):
                Y = real_OB[i, j]
                Yhat1 = predicted_OB[i, j]
                Yhat2 = predicted_OB[i, k]
                if ID1 == ID2:
                    OB_pvalue = 1
                else:
                    OB_pvalue = DM(Y, Yhat1, Yhat2, norm="smape")
                    
                OB_pvalues[i, j, k] = OB_pvalue        

            # Compute the DM test on the Prices
            if (not model_wrapper_1.predict_order_books) and (not model_wrapper_2.predict_order_books):
                Y = real_prices[i, j]
                Yhat1 = predicted_prices[i, j]
                Yhat2 = predicted_prices[i, k]                
                if ID1 == ID2:
                    prices_pvalue = 1
                else:
                    prices_pvalue = DM(Y, Yhat1, Yhat2, norm="mae")
                    
                prices_pvalues[i, j, k] = prices_pvalue

with matplotlib.rc_context({ "text.usetex" : True,
                             "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                             "font.family" : ""}):
    plt.close("all")    
    plot_DM_tests(
        prices_pvalues, countries=countries, 
        IDs=np.arange(1, 8), label="prices")
    plt.show()

with matplotlib.rc_context({ "text.usetex" : True,
                             "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                             "font.family" : ""}):
    plt.close("all")    
    plot_DM_tests(
    OB_pvalues, countries=countries, IDs=np.arange(1, 8), label="Order Books")
    plt.show()
    

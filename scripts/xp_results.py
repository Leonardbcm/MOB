%load aimport
import itertools, pandas, numpy as np, datetime, matplotlib.pyplot as plt, time

from src.models.spliter import MySpliter
from src.models.torch_wrapper import OBNWrapper
from src.models.torch_models.weight_initializers import *
import src.models.parallel_scikit as ps
from src.analysis.utils import *

"""
XP results and analysis file
"""

####### Re-create the configurations
skip_connections = [True, False]
#use_order_books = [True, False]
use_order_books = [False]
separate_optims = [True, False]
#order_book_sizes = [20, 50, 100, 250]
order_book_sizes = [20, 50]
countries = ["FR", "DE", "BE", "NL"]
datasets = ["Lyon", "Munich", "Bruges", "Lahaye"]

combinations = list(itertools.product(
    skip_connections, use_order_books, separate_optims, order_book_sizes))
spliter = MySpliter(365, shuffle=False)

# Recreate all model wrapper objects
model_wrappers = get_model_wrappers(combinations, countries, datasets, spliter)

# Recompute results
results = pandas.DataFrame(
    columns=["country", "separate_optim", "skip_connection",
             "use_order_book", "order_book_size",  "stopped_epoch",
             "val_mae", "val_dae", "val_smape","val_ACC",
             "test_mae", "test_dae", "test_smape","test_ACC"])
for i, model_wrapper in enumerate(model_wrappers):
    line = model_wrapper.compute_metrics()
    stopped_epoch = model_wrapper.get_stopped_epoch(model_wrapper.highest_version)
    if line != {}:
        line["stopped_epoch"] = stopped_epoch
        line = pandas.DataFrame(line, index=[i])
        results = pandas.concat([results, line])

        # Ensure types
        results.separate_optim = results.separate_optim.astype("boolean")
        results.skip_connection = results.skip_connection.astype("boolean")
        results.use_order_book = results.use_order_book.astype("boolean")        

# Filter models that have not run
df = results[results.stopped_epoch > 0]
        
# Get log path
model_wrapper = model_wrappers[df.index[0]]
model_wrapper.plot_forecasts()
model_wrapper.logs_path

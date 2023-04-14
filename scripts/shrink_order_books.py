%load aimport

import os, datetime, numpy as np, pandas, matplotlib, matplotlib.pyplot as plt
import itertools, time
from torch.utils.data import DataLoader
from src.analysis.utils import load_real_prices
from src.analysis.shrink_order_books_utils import OB_to_csv

from src.models.torch_models.ob_datasets import OrderBookDataset
from src.models.spliter import MySpliter
from src.models.torch_wrapper import OBNWrapper
from src.models.torch_models.torch_solver import BatchPFASolver
from src.euphemia.order_books import *
from src.euphemia.solvers import *
from src.euphemia.ploters import get_ploter

base_folder = os.environ["MOB"]

country = "BE"
data_folder = os.path.join(base_folder, "curves")
df = load_real_prices(country)
datetimes = df.index

##################### Create an order book dataset and a solver
obd = OrderBookDataset(country, data_folder, df.index, 20, requires_grad=False,
                       real_prices=df, coerce_size=True)
obd[0][0].shape
loader = DataLoader(obd, batch_size=10, num_workers=os.cpu_count())

# Get a batch
batch, batch_idx = next(iter(loader))

########## Comparison plot
idt = 0
country = "DE"
df = load_real_prices(country)
fig, ax = plt.subplots(1, figsize=(19.2, 10.8))

OBref = LoadedOrderBook(df.index[idt], os.path.join(data_folder, country))
solverref = MinDual(OBref)
solverref.solve("dual_derivative_heaviside")
ploterref = get_ploter(OBref)
ploterref.display(ax_=ax, colors="k", labels="original order book")

OBtries = [20, 50, 100, 250]
colors = ["r", "m", "b", "c", "g"]
N = 1000
for j, OBs in enumerate(OBtries):
    obd = OrderBookDataset(country, data_folder, df.index, OBs, requires_grad=False,
                           real_prices=df)
    OBshrink = SimpleOrderBook(TorchOrderBook(obd[idt][0]).orders)
    solver = MinDual(OBshrink)
    res = solver.solve("dual_derivative_heaviside")
    print(OBs, res)
    
    ploter = get_ploter(OBshrink)
    ploter.display(ax_=ax, colors=colors[j], labels=f"shrinked to {OBs}")
ax.legend()
plt.show()

############ Small XP
country = "NL"
df = load_real_prices(country)
OBtries = [20, 30, 50, 100, 500]
N = 1000
results = np.zeros((N, len(OBtries)))
for i, dt in enumerate(range(N)):
    for j, OBs in enumerate(OBtries):
        obd = OrderBookDataset(country,
                               data_folder, df.index, OBs, requires_grad=False,
                               real_prices=df)
        OBshrink = SimpleOrderBook(TorchOrderBook(obd[dt][0]).orders)
        solver = MinDual(OBshrink)
        results[i, j] = solver.solve("dual_derivative_heaviside")

############ All datasets, all days
df = load_real_prices("FR")
datetimes = df.index[:20]
countries = ["FR", "DE", "BE", "NL"]
OBsizes = [20, 50, 100, 250]
niters = [10, 25, 40]
ks = [20, 40, 60, 80]
batch_size = 30
results = pandas.DataFrame(
    columns=["period_start_time", "country","OBs", "niter", "k", "price", "time", "real_price", "error"])
for country in countries:
    print(f"Country {country}")
    df = load_real_prices(country)
    for OBs in OBsizes:
        print(f"OBs {OBs}")
        obd = OrderBookDataset(country, data_folder, datetimes, OBs,
                               requires_grad=False,real_prices=df, coerce_size=True)
        loader = DataLoader(obd, batch_size=batch_size*24)
        OB = np.zeros((len(df.index), OBs, 3))
        for i, (batch, idx) in enumerate(loader):
            print(f"Batch {i}")            
            OB[idx.numpy(), :, :] = batch.detach().numpy()
            real_price = df.loc[datetimes[idx.numpy()], "price"].values
            for (niter, k) in itertools.product(niters, ks):                
                solver = BatchPFASolver(niter=niter, k=k)
                start = time.time()
                pstars = solver(batch).reshape(-1).detach().numpy()
                stop = time.time()
                
                lines = pandas.DataFrame(
                    {"price":pstars, "country":country, "real_price":real_price,
                     "period_start_time" : df.index[idx.numpy()],
                     "error":np.abs(real_price-pstars),
                     "OBs":OBs, "niter":niter, "k":k, "time":stop - start})
                results = pandas.concat([results, lines], ignore_index=True)

        np.save(os.path.join(base_folder, "curves", f"{country}_{OBs}.npy"), OB)


belgium = results.loc[results.country == "BE"].copy()        
df = load_real_prices("BE")        
n_batches = 1 + len(df.index) // (batch_size*24)
last_batch_size = len(df.index) - (n_batches - 1) * (batch_size*24)
current = 0
belgium.index = np.arange(len(belgium.index))
for OB in OBsizes:
    for i  in range(n_batches):
        if i == (n_batches - 1):
            indices = [24 * batch_size *i + j for j in range(last_batch_size)]
        else:
            indices = [24 * batch_size * i + j for j in range(24 * batch_size)]
        datetimes = list(df.index[indices].values)            
        for (niter, k) in itertools.product(niters, ks):
            belgium.loc[current:current+len(datetimes)-1, "period_start_time"] = datetimes
            current += len(datetimes)

france.loc[:, "error"] = np.abs(df.loc[france.period_start_time.values, "price"].values - france.price.values)
france.groupby(["OBs", "niter", "k"]).error.mean()

germany.loc[:, "error"] = np.abs(df.loc[germany.period_start_time.values, "price"].values - germany.price.values)
germany.groupby(["OBs", "niter", "k"]).error.mean()

belgium.loc[:, "error"] = np.abs(df.loc[belgium.period_start_time.values, "price"].values - belgium.price.values)
belgium.groupby(["OBs", "niter", "k"]).error.mean()

country = "DE"  
df = load_real_prices(country)
datetimes = df.index
dates = df.period_start_date        
OBs = 20
OB = np.load(os.path.join(base_folder, "curves", f"{country}_{OBs}.npy"))
res = OB_to_csv(OB, datetimes)
res.to_csv(os.path.join(base_folder, "curves", f"{country}_{OBs}.csv"),
           index_label="period_start_date")
res = pandas.read_csv(os.path.join(base_folder, "curves", f"{country}_{OBs}.csv"),
                      index_col="period_start_date")

###### SAVE DATASETS
data_folder = os.path.join(base_folder, "curves")

countries = ["FR", "DE", "BE", "NL"]
#OBsizes = [20, 50, 100, 250]
OBsizes = [100, 250]
batch_size = 30
country = "BE"
print(f"Country {country}")
df = load_real_prices(country)
datetimes = df.index
for OBs in OBsizes:
    print(f"OBs {OBs}")
    obd = OrderBookDataset(country, data_folder, datetimes, OBs,
                           requires_grad=False,real_prices=df, coerce_size=True)
    OB = np.zeros((len(df.index), OBs, 3))
    for i, dt in enumerate(datetimes):
        OB[i, :, :] = obd[i][0].detach().numpy()
        
    np.save(os.path.join(base_folder, "curves", f"{country}_{OBs}.npy"), OB)
    res = OB_to_csv(OB, datetimes)
    res.to_csv(os.path.join(base_folder, "curves", f"{country}_{OBs}.csv"),
               index_label="period_start_date")

###### Correct summer to winter
summer_to_winter = [datetime.date(2016, 10, 30),
                    datetime.date(2017, 10, 29),
                    datetime.date(2018, 10, 28),
                    datetime.date(2019, 10, 27),
                    datetime.date(2020, 10, 25),
                    datetime.date(2021, 10, 31)]
# Load dataset
spliter = MySpliter(365, shuffle=False)
country = "BE"
dataset = "Bruges"
for OBs in [20, 50, 100, 250]:
    model_wrapper = OBNWrapper("TEST", dataset, country=country, spliter=spliter,
                               use_order_books=True, order_book_size=OBs,
                               separate_optim=True)

    # Load OB dataframe
    order_book_path = os.path.join(
        os.environ["MOB"],"curves",
        f"{model_wrapper.country}_{model_wrapper.order_book_size}.csv")
    OB = pandas.read_csv(order_book_path)            
    OB.index = [datetime.datetime.strptime(
        d, "%Y-%m-%d") for d in OB.period_start_date]
    OB.drop(columns="period_start_date", inplace=True)

    # Load prices
    df = load_real_prices(country)
    
    # Create OB loader
    obd = OrderBookDataset(country, data_folder, df.index, OBs, real_prices=df)
    for day in summer_to_winter[-2:]:
        # Compute the index and columns of hour 2
        date_time = datetime.datetime(day.year, day.month, day.day, 2)
        columns = [f"OB_2_{V}_{i}" for V in ["V", "Po", "P"] for i in range(OBs)]
        past_columns = [f"OB_2_{V}_{i}_past_1" for V in ["V", "Po", "P"]
                        for i in range(OBs)]    
        idx = np.where(df.index == date_time)[0][0]
        past_day = day + datetime.timedelta(hours=24)
        
        # Load Data for hour 2
        data = obd[idx][0].detach().numpy().reshape(-1, order='F')
        
        # Replace data of hour 2 in the dataset
        OB.loc[pandas.to_datetime(day), columns] = data.copy()
        
        # Replace past data of hour 2
        OB.loc[pandas.to_datetime(past_day), past_columns] = data.copy()
    
    OB.to_csv(order_book_path)

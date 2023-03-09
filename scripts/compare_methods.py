%load_ext autoreload
%autoreload 1
%aimport src.analysis.compare_methods_utils 

import os, datetime, numpy as np, pandas, matplotlib, matplotlib.pyplot as plt
from src.euphemia.order_books import LoadedOrderBook, TorchOrderBook
from src.euphemia.solvers import MinDual, TorchSolution
from src.euphemia.ploters import get_ploter
from src.analysis.compare_methods_utils import it_results

from torch.utils.data import DataLoader
from src.models.obn.ob_datasets import OrderBookDataset
from src.models.obn.torch_solver import BatchPFASolver
################## LOAD from real data AND compare solutions for 1 day
base_folder = os.environ["MOB"]
data_folder = os.path.join(base_folder, "HOURLY")
date_time = datetime.datetime(2016, 10, 30, 3)

OB = LoadedOrderBook(date_time, data_folder)
solver = MinDual(OB)
ploter = get_ploter(OB, solver)

# Display Graphic Solution of the following methods
ploter.arrange_plot("display", "dual_function",
                    ["dual_derivative", {"method" : "piecewise"}],
                    ["dual_derivative", {"method" : "sigmoid"}],
                    ["dual_derivative", {"method" : "generic_heaviside"}],
                    ["dual_derivative", {"method" : "generic_sigmoid"}])

# Use a torch solver to compute the real solution using Dichotomic search
torch_book = TorchOrderBook(OB.orders)
torch_solver = TorchSolution(torch_book)
pstar = torch_solver.solve()

################## Load real prices + datetimes
filename = "real_prices.csv"
path = os.path.join(base_folder, filename)
df = pandas.read_csv(path)
df.period_start_time = [datetime.datetime.strptime(d, "%Y-%m-%dT%H:00:00")
                        for d in df.period_start_time]
df.period_start_date = [d.date() for d in df.period_start_time]
df.set_index("period_start_time", inplace=True, drop=True)
df = df.head(10)

################## LOAD from real data AND compare solutions for all days
datetimes = df.index
ks = [1, 10, 100, 1000]
OB = LoadedOrderBook(datetimes[0], data_folder)
methods = MinDual(OB).solve_all_methods(ks=ks)[1]
methods += TorchSolution(TorchOrderBook(OB.orders)).solve_all_methods(ks=ks)[1]
results = pandas.DataFrame(columns=methods, index=datetimes)
for date_time in datetimes:
    OB = LoadedOrderBook(date_time, data_folder)
    solver = MinDual(OB)
    res, methods_ = solver.solve_all_methods(ks=ks)
    results.loc[date_time, methods_] = res
    
    torch_book = TorchOrderBook(OB.orders)
    torch_solver = TorchSolution(torch_book)
    
    res, methods_ = torch_solver.solve_all_methods(ks=ks)
    results.loc[date_time, methods_] = res            
    results.loc[date_time, "real_price"] = df.loc[date_time, "price"]

filename = "solvers.csv"
path = os.path.join(base_folder, filename)
results.to_csv(path)

real_prices = results.loc[:, "real_price"]
maes = {}
for method in methods:
    forecast = results.loc[:, method]
    maes[method] = np.abs(forecast - real_prices).mean()

filename = "k_results.csv"
path = os.path.join(base_folder, filename)
pandas.DataFrame(
    columns=list(maes.keys()),
    index=[1],
    data=np.array(list(maes.values())).reshape(1, -1)).to_csv(path, index=False)    

# Errors of the order of 1 or 2 cts for the sigmoid approximation, that is
# sufficiently small!
############## Use the torch Layer to compute prices batch per batch
# Batch Solving for several niter
order_book_dataset = OrderBookDataset(data_folder, df.index, 3000)
loader = DataLoader(order_book_dataset, batch_size=30*24)
niters = range(5, 35, 5)
results = pandas.DataFrame(columns=[f"BatchSolver{niter}" for niter in niters],
                           index=datetimes)
results.loc[:, "real_price"] = df.price
for niter in niters:
    column = f"BatchSolver{niter}"
    solver = BatchPFASolver(niter=niter, k=100)
    results.loc[:, column] = np.zeros(len(df.index), dtype=np.float32)    
    for batch, idx in loader:
        pstars = solver(batch).reshape(-1)
        results.loc[results.index[idx.numpy()], column] = pstars.detach().numpy()

filename = "torch_solvers.csv"
path = os.path.join(base_folder, filename)
results.to_csv(path)        

maes = {}
for niter in range(5, 35, 5):
    column = f"BatchSolver{niter}"
    maes[column] = np.abs(results.real_price - results.loc[:, column]).mean()

filename = "niter_results.csv"
path = os.path.join(base_folder, filename)
pandas.DataFrame(
    columns=list(maes.keys()),
    index=[1],
    data=np.array(list(maes.values())).reshape(1, -1)).to_csv(path, index=False)

################### Plot results
with matplotlib.rc_context({ "text.usetex" : True,
                             "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                             "font.family" : ""}):
    plt.close("all")
    it_results()
    plt.show()

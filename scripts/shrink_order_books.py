%load aimport

import os, datetime, numpy as np, pandas, matplotlib, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.analysis.utils import load_real_prices

from src.models.torch_models.ob_datasets import OrderBookDataset, DirectOrderBookDataset
from src.models.torch_models.torch_solver import BatchPFASolver
from src.euphemia.order_books import *
from src.euphemia.solvers import *
from src.euphemia.ploters import get_ploter

base_folder = os.environ["MOB"]
data_folder = os.path.join(base_folder, "HOURLY")
df = load_real_prices()
datetimes = df.index

##################### Create an order book dataset and a solver
obd = OrderBookDataset(data_folder, df.index, 50, requires_grad=False,
                       real_prices=df)
obd[0][0].shape
loader = DataLoader(obd, batch_size=10, num_workers=os.cpu_count())

# Get a batch
batch, batch_idx = next(iter(loader))

######## Compare
############ Small XP
fig, ax = plt.subplots(1, figsize=(19.2, 10.8))
ploterref.display(ax_=ax, colors="k", labels="original order book")

OBref = LoadedOrderBook(df.index[0], data_folder)
solverref = MinDual(OBref)
solverref.solve("dual_derivative_heaviside")
ploterref = get_ploter(OBref)

OBtries = [20, 30, 50, 100, 500]
colors = ["r", "m", "b", "c", "g"]
N = 1000
for j, OBs in enumerate(OBtries):
    obd = OrderBookDataset(data_folder, df.index, OBs, requires_grad=False,
                           real_prices=df)
    OBshrink = SimpleOrderBook(TorchOrderBook(obd[dt][0]).orders)
    solver = MinDual(OBshrink)
    res = solver.solve("dual_derivative_heaviside")
    print(OBs, res)
    
    ploter = get_ploter(OBshrink)
    ploter.display(ax_=ax, colors=colors[j], labels=f"shrinked to {OBs}")
ax.legend()
plt.show()

############ Small XP
OBtries = [20, 30, 50, 100, 500]
N = 1000
results = np.zeros((N, len(OBtries)))
for i, dt in enumerate(range(N)):
    for j, OBs in enumerate(OBtries):
        obd = OrderBookDataset(data_folder, df.index, OBs, requires_grad=False,
                               real_prices=df)
        OBshrink = SimpleOrderBook(TorchOrderBook(obd[dt][0]).orders)
        solver = MinDual(OBshrink)
        results[i, j] = solver.solve("dual_derivative_heaviside")
    

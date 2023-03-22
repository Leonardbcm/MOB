%load aimport

import os, datetime, numpy as np, pandas, matplotlib, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.analysis.utils import load_real_prices

from src.models.torch_models.ob_datasets import OrderBookDataset
from src.models.torch_models.torch_solver import BatchPFASolver
from src.euphemia.order_books import *
from src.euphemia.ploters import get_ploter

base_folder = os.environ["MOB"]
data_folder = os.path.join(base_folder, "HOURLY")
df = load_real_prices()
datetimes = df.index

##################### Create an order book dataset and a solver
order_book_dataset = OrderBookDataset(data_folder, df.index, 3000)
loader = DataLoader(order_book_dataset, batch_size=30*24)
solver = BatchPFASolver(niter=30, k=100)

# Get a batch
batch, idx = next(iter(loader))

##################### Compare 2 loading methods
date_time = datetimes[0]
fig, ax = plt.subplots(1, figsize=(19.2, 10.8))

OB1 = LoadedOrderBook(date_time, data_folder, volume_lines=False)
ploter = get_ploter(OB1)
ploter.display(ax_=ax, colors="r", labels="Without Storing prices")

OB2 = LoadedOrderBook(date_time, data_folder, volume_lines=True)
ploter = get_ploter(OB2)
ploter.display(ax_=ax, colors="b", labels="Storing prices")
plt.legend()
plt.show()

###################### Direct tensors
datetime_str = datetime.datetime.strftime(date_time, "%Y-%m-%d_%Hh")
supply_volumes = np.load(os.path.join(
    data_folder, f"{datetime_str}_supply_volumes.npy"))
supply_prices = np.load(os.path.join(
    data_folder, f"{datetime_str}_supply_prices.npy"))
supply_price_ranges = np.copy(supply_prices)

supply_volumes[1:] = supply_volumes[1:] - supply_volumes[:-1]
supply_price_ranges[1:] = supply_price_ranges[1:] - supply_price_ranges[:-1]

# Filter null volumes since they bring nothing
indices = np.where(supply_volumes > 0)[0]
supply_volumes = supply_volumes[indices]
supply_price_ranges = supply_price_ranges[indices]
# First order is step
supply_price_ranges[0] = 0

supply_prices = supply_prices[indices - 1]
supply_prices[0] = -500

supply = SimpleOrderBook(OB1.supply)
assert (supply_volumes == supply.volumes).all()
assert (supply_price_ranges == supply.prices).all()
assert (supply_prices == supply.p0s).all()

######### Demand
datetime_str = datetime.datetime.strftime(date_time, "%Y-%m-%d_%Hh")
demand_volumes = np.load(os.path.join(
    data_folder, f"{datetime_str}_demand_volumes.npy"))
demand_prices = np.load(os.path.join(
    data_folder, f"{datetime_str}_demand_prices.npy"))
demand_price_ranges = np.copy(demand_prices)

demand_volumes[:-1] = demand_volumes[:-1] - demand_volumes[1:]
demand_price_ranges[:-1] = demand_price_ranges[:-1] - demand_price_ranges[1:]

# Filter null volumes since they bring nothing
indices = np.where(demand_volumes > 0)[0]
demand_volumes = demand_volumes[indices]
demand_price_ranges = demand_price_ranges[indices]
# Last order is step
demand_price_ranges[-1] = 0
demand_prices = np.concatenate((demand_prices, [3000]))[indices + 1]

demand = SimpleOrderBook(OB1.demand)
assert (demand_volumes[::-1] == demand.volumes).all()
assert (demand_price_ranges[::-1] == demand.prices).all()
assert (demand_prices[::-1] == demand.p0s).all()

####### Concat and reshape
size_s = len(supply_volumes)
size_d = len(demand_volumes)
size =  size_s + size_d
res = np.zeros((size, 3))

res[:size_s, 0] = supply_volumes
res[size_s:, 0] = - demand_volumes

res[:size_s, 1] = supply_prices
res[size_s:, 1] = demand_prices

res[:size_s, 2] = supply_price_ranges
res[size_s:, 2] = demand_price_ranges

###### To torch & final reshape
torch_book = torch.tensor(res, dtype=torch.float32)
torch_book.reshape(1, -1, 3)

import pandas as pd, numpy as np, torch, datetime, pandas, os
from torch.utils.data import Dataset
from pytorch_lightning import LightningModule
from src.euphemia.order_books import LoadedOrderBook, TorchOrderBook
from src.euphemia.order_books import SimpleOrderBook
from src.models.torch_models.torch_solver import *
from src.euphemia.solvers import *

class EPFDataset(Dataset):
    """
    Helps constructing DataLoaders
    """
    def __init__(self, X, Y=None, dtype=torch.float32):
        self.dtype = dtype
        self.X = torch.tensor(X.astype(float), dtype = dtype)

        if Y is not None:
            self.Y = torch.tensor(Y.astype(float), dtype = dtype)
        else:
            self.Y = None
            
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.Y is not None:
            return (self.X[idx, :], self.Y[idx, :])
        else:
            return self.X[idx, :]

    
class OrderBookDataset(Dataset):
    def __init__(self, country, data_folder, datetimes, OBs,
                 real_prices = None,
                 niter=30, pmin=-500, pmax=3000, k=100,
                 coerce_size=True, requires_grad=True, dtype=torch.float32):
        self.country = country
        self.data_folder_ = data_folder
        self.data_folder = os.path.join(data_folder, country)
        self.real_prices = real_prices
        self.datetimes = datetimes
        self.OBs = OBs
        self.requires_grad = requires_grad
        self.dtype = dtype
        self.coerce_size_ = coerce_size

        self.ns = int(self.OBs / 2)
        self.nd = int(self.OBs / 2)
        if (self.OBs % 2) != 0:            
            self.nd += 1
            
        self.nd -= 3
        self.ns -= 3

        self.ns_before = int(self.ns / 2)
        self.ns_after = int(self.ns / 2)
        if (self.ns % 2) != 0:
            self.ns_after += 1

        self.nd_before = int(self.nd / 2)
        self.nd_after = int(self.nd / 2)
        if (self.nd % 2) != 0:
            self.nd_after += 1

        self.pmin = pmin
        self.pmax = pmax
        self.k = k
        self.niter = niter        
        
    def __len__(self):
        return len(self.datetimes)

    def __getitem__(self, idx):
        date_time = self.datetimes[idx]
        date_time = datetime.datetime(date_time.year, date_time.month,
                                      date_time.day, date_time.hour)
        order_book = LoadedOrderBook(date_time, self.data_folder)
        
        if self.coerce_size_:
            try:
                order_book = self.shrink(order_book, idx)
            except:
                raise(Exception("Failed for", date_time))
            
        order_book = TorchOrderBook(
            order_book.orders, requires_grad=self.requires_grad)
        data = order_book.data

        if (self.coerce_size_) and (data.shape[1] < self.OBs):
            data = self.extend(data)
            raise(Exception("Failed for", date_time, f" : shape of data is {data.shape[1]}"))
        else:
            data = data.reshape(-1, 3)
        
        return data, torch.tensor(idx)

    def shrink(self, OB, idx):
        if self.real_prices is None:
            self.solver = MinDual(OB, pmin=self.pmin, pmax=self.pmax)            
            pstar = self.solver.solve("dual_derivative_heaviside")
        else:
            pstar = self.real_prices.loc[self.datetimes[idx], "price"]

        # Sum limit step orders that have the same price
        supply_limit_indices = np.where(np.logical_and(
            OB.p0s == OB.pmin,
            np.logical_and(OB.prices == 0, OB.signs == 1)))[0]
        supply_limit_volume = OB.volumes[supply_limit_indices].sum()
        supply_limit_order = StepOrder("Supply",OB.pmin,supply_limit_volume)

        demand_limit_indices = np.where(np.logical_and(
            OB.p0s == OB.pmax,
            np.logical_and(OB.prices == 0, OB.signs == -1)))[0]
        demand_limit_volume = OB.volumes[demand_limit_indices].sum()
        demand_limit_order = StepOrder("Demand",OB.pmax,demand_limit_volume)

        indices_to_remove = np.concatenate((
            supply_limit_indices, demand_limit_indices))
        mask = np.ones(OB.n, dtype=bool)
        mask[indices_to_remove] = False
        OB.orders = OB.orders[mask]
            
        ####### Supply
        supply_orders = [supply_limit_order]
        OBs = SimpleOrderBook(OB.supply)

        ### Before
        # Sort orders by closeness to pstar
        supply_before = OBs.orders[((pstar - OBs.p0s) >= 0)]
        supply_orders_before = list(supply_before[-self.ns_before:])
        
        # If there is for supply orders before intersection, summarize them
        if len(supply_before) > self.ns_before:
            remaining_supply_before=SimpleOrderBook(supply_before[:-self.ns_before])
            supply_orders += [remaining_supply_before.sum("Supply")]
        else:
            # Otherwise, fill to reach the correct number + 1
            supply_orders_before += [
                StepOrder("Supply", pstar, 0)
                for i in range(self.ns_before - len(supply_orders_before) + 1)]
        supply_orders += supply_orders_before

        ### After        
        supply_after = OBs.orders[((pstar - OBs.p0s) < 0)]
        supply_orders_after = list(supply_after[:self.ns_after])

        if len(supply_after) > self.ns_after:
            remaining_supply_after = SimpleOrderBook(supply_after[self.ns_after:])
            supply_orders_after += [remaining_supply_after.sum("Supply")]
        else:
            # Fill at the begining!
            supply_orders_after = [StepOrder("Supply", pstar, 0) for i in range(self.ns_after - len(supply_orders_after) + 1)] + supply_orders_after
        supply_orders += supply_orders_after
        
        ######## Demand
        # Extract the limit order
        demand_orders = [demand_limit_order]
        OBd = SimpleOrderBook(OB.demand)

        ### Before
        demand_before = OBd.orders[((pstar - OBd.p0s) < 0)]
        demand_orders_before = list(demand_before[-self.nd_before:])
        
        if len(demand_before) > self.nd_before:
            remaining_demand_before=SimpleOrderBook(demand_before[:-self.nd_before])
            demand_orders += [remaining_demand_before.sum("Demand")]
        else:
            demand_orders_before += [
                StepOrder("Demand", pstar, 0)
                for i in range(self.nd_before - len(demand_orders_before) + 1)]
        demand_orders += demand_orders_before

        ### After
        demand_after = OBd.orders[((pstar - OBd.p0s) >= 0)]
        demand_orders_after = list(demand_after[:self.nd_after])
        
        if len(demand_after) > self.nd_after:
            remaining_demand_after = SimpleOrderBook(demand_after[self.nd_after:])
            demand_orders_after += [remaining_demand_after.sum("Demand")]
        else:
            demand_orders_after = [StepOrder("Demand", pstar, 0) for i in range(self.nd_after - len(demand_orders_after) + 1)] + demand_orders_after
        demand_orders += demand_orders_after            
            
        OB_shrinked = SimpleOrderBook(np.array(supply_orders + demand_orders))
        return OB_shrinked

    def extend(self, data):
        coerced_order_book = torch.zeros((self.OBs, 3), dtype=self.dtype)
        coerced_order_book[:data.shape[1], :] = data
        return coerced_order_book
    
    
class DirectOrderBookDataset(OrderBookDataset):
    """
    Directly Load Data, no time to create order objects
    """
    def __init__(self, data_folder, datetimes, OBs, requires_grad=True,
                 coerce_size=True, dtype=torch.float32):
        OrderBookDataset.__init__(self, data_folder, datetimes, OBs,
                                  requires_grad=requires_grad,
                                  coerce_size=coerce_size, dtype=torch.float32)

    def __getitem__(self, idx):
        date_time = self.datetimes[idx]
        datetime_str = datetime.datetime.strftime(date_time, "%Y-%m-%d_%Hh")        

        # Supply part
        supply_volumes = np.load(os.path.join(
            self.data_folder, f"{datetime_str}_supply_volumes.npy"))
        supply_prices = np.load(os.path.join(
            self.data_folder, f"{datetime_str}_supply_prices.npy"))
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

        # Demand        
        demand_volumes = np.load(os.path.join(
            self.data_folder, f"{datetime_str}_demand_volumes.npy"))
        demand_prices = np.load(os.path.join(
            self.data_folder, f"{datetime_str}_demand_prices.npy"))        

        demand_volumes = np.load(os.path.join(
            self.data_folder, f"{datetime_str}_demand_volumes.npy"))
        demand_prices = np.load(os.path.join(
            self.data_folder, f"{datetime_str}_demand_prices.npy"))
        demand_price_ranges = np.copy(demand_prices)

        demand_volumes[:-1] = demand_volumes[:-1] - demand_volumes[1:]
        demand_price_ranges[:-1] = demand_price_ranges[:-1]- demand_price_ranges[1:]

        # Filter null volumes since they bring nothing
        indices = np.where(demand_volumes > 0)[0]
        demand_volumes = demand_volumes[indices]
        demand_price_ranges = demand_price_ranges[indices]
        
        # Last order is step
        pmax = demand_price_ranges[-1]
        demand_price_ranges[-1] = 0
        demand_prices = np.concatenate((demand_prices, [pmax]))[indices + 1]        

        size_s = len(supply_volumes)
        size_d = len(demand_volumes)
        size =  size_s + size_d
        res = np.zeros((size, 3))
        
        res[:size_s, 0] = supply_volumes
        res[size_s:, 0] = - demand_volumes[::-1]
        
        res[:size_s, 1] = supply_prices
        res[size_s:, 1] = demand_prices[::-1]
        
        res[:size_s, 2] = supply_price_ranges
        res[size_s:, 2] = demand_price_ranges[::-1]

        ###### To torch & final reshape
        torch_book = torch.tensor(res, dtype=torch.float32)
        torch_book = torch_book.reshape(1, -1, 3)

        if self.coerce_size_:
            torch_book = self.coerce_size(torch_book)

        torch_book.requires_grad_(self.requires_grad) 
        return torch_book, torch.tensor(idx)
        

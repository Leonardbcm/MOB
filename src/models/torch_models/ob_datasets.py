import pandas as pd, numpy as np, torch, datetime, pandas, os
from torch.utils.data import Dataset
from pytorch_lightning import LightningModule
from src.euphemia.order_books import LoadedOrderBook, TorchOrderBook
from src.euphemia.order_books import SimpleOrderBook

class EPFDataset(Dataset):
    """
    Helps constructing DataLoaders
    """
    def __init__(self, X, Y=None, dtype=torch.float32, N_OUTPUT=24):
        self.dtype = dtype
        self.N_OUTPUT = N_OUTPUT
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
    def __init__(self, data_folder, datetimes, OBs, coerce_size=True,
                 requires_grad=True, dtype=torch.float32):
        self.data_folder = data_folder
        self.datetimes = datetimes
        self.OBs = OBs
        self.requires_grad = requires_grad
        self.dtype = dtype
        self.coerce_size_ = coerce_size
        
    def __len__(self):
        return len(self.datetimes)

    def __getitem__(self, idx):
        date_time = self.datetimes[idx]
        loaded_order_book = LoadedOrderBook(date_time, self.data_folder)
        torch_order_book = TorchOrderBook(
            loaded_order_book.orders, requires_grad=self.requires_grad)
        data = torch_order_book.data

        if self.coerce_size_:
            data = self.coerce_size(data)
            
        return data, torch.tensor(idx)
        
    def coerce_size(self, data):
        """
        Given an order book of size 1 x S x 3, returns another order book
        of shape 1 x OBs x 3.

        This transformation shall not modify equilibrium.
        In case there are not enough orders, We add orders with V=0 to fill the OB.
        The case where there are too much orders is not handled.
        """
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
        

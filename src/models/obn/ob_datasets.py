import pandas as pd, numpy as np, torch, datetime, pandas
from torch.utils.data import Dataset
from pytorch_lightning import LightningModule
from src.euphemia.order_books import LoadedOrderBook, TorchOrderBook

class OrderBookDataset(Dataset):
    def __init__(self, data_folder, datetimes, OBs, dtype=torch.float32):
        self.data_folder = data_folder
        self.datetimes = datetimes
        self.OBs = OBs
        self.dtype = dtype
        
    def __len__(self):
        return len(self.datetimes)

    def __getitem__(self, idx):
        date_time = self.datetimes[idx]
        loaded_order_book = LoadedOrderBook(date_time, self.data_folder)
        torch_order_book = TorchOrderBook(loaded_order_book.orders)
        
        return self.coerce_size(torch_order_book.data), torch.tensor(idx)
        

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
    

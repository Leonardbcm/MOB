import numpy as np, torch, datetime, os
from src.euphemia.orders import *

class OrderBook(object):
    """
    An order Book is a collection of Orders
    """
    def __init__(self, orders):
        self.orders = np.array(orders)        

    @property
    def n(self):
        return len(self.orders)
    
    @property    
    def signs(self):
        return np.array([o.sign for o in self.orders])

    @property    
    def directions(self):
        return np.array([o.direction for o in self.orders])

    @property    
    def supply(self):
        return np.array([o for o in self.orders if o.is_supply])

    @property    
    def demand(self):
        return np.array([o for o in self.orders if o.is_demand])
        
    def __repr__(self):
        s = f"{type(self).__name__}(\n"
        for o in self.orders:
            s += "\t" + str(o) + "\n"
        return s + ")"
    
    
class SimpleOrderBook(OrderBook):
    """
        Order Book containing only simple orders (step or linear)
    """
    def __init__(self, orders):
        OrderBook.__init__(self, orders)      

    @property    
    def volumes(self):
        return np.array([o.V for o in self.orders])

    @property    
    def prices(self):
        return np.array([o.P for o in self.orders])

    @property
    def p0s(self):
        return np.array([o.p0 for o in self.orders])

    @property
    def pmin(self):
        prices = self.prices
        p0s = self.p0s
        pmin = min(min(p0s), min(p0s + prices))
        return pmin

    @property
    def pmax(self):
        prices = self.prices
        p0s = self.p0s
        pmax = max(max(p0s), max(p0s + prices))
        return pmax

    @property
    def pmin_supply(self):
        return SimpleOrderBook(self.supply).pmin

    @property
    def pmax_supply(self):
        return SimpleOrderBook(self.supply).pmax
    
    @property
    def pmin_demand(self):
        return SimpleOrderBook(self.demand).pmin

    @property
    def pmax_demand(self):
        return SimpleOrderBook(self.demand).pmax
            
    @property
    def vmin(self):
        return min(self.volumes)

    @property
    def vmax(self):
        return max(self.volumes)

    @property
    def vsum(self):
        sum_demand = sum([o.V for o in self.orders if o.is_demand])
        sum_supply = sum([o.V for o in self.orders if o.is_supply])
        return max([sum_demand, sum_supply])
    
    def price_range(self, pmin, pmax, step=0.01):
        return np.arange(pmin, pmax + step/2, step)

    def sum(self, direction):
        # Only 1 step order in order book!
        if self.n == 1:
            return self.orders[0]
        
        v = self.volumes.sum()
        p1 = self.orders[0].p0
        p2 = self.orders[-1].p0  + self.orders[-1].P

        if p1 == p2:
            order = StepOrder(direction, p2, v)
        else:
            order = LinearOrder(direction, p1, p2, v)
        return order
    
    def _curves_fit(self, step=0.01):
        pmin_supply = self.pmin_supply
        pmax_supply = self.pmax_supply            
        price_range_supply=self.price_range(pmin_supply, pmax_supply, step=step)
        supply = np.zeros(len(price_range_supply))
        for os in self.supply:
            supply += os.differential(pmin_supply, pmax_supply, step)

        pmin_demand = self.pmin_demand
        pmax_demand = self.pmax_demand            
        price_range_demand = self.price_range(pmin_demand, pmax_demand, step=step)
        demand = np.zeros(len(price_range_demand))
        for od in self.demand:
            demand += od.differential(pmin_demand, pmax_demand, step)

        return ((np.cumsum(supply), price_range_supply),
                (np.cumsum(demand), price_range_demand))        

    def _curves_not_fit(self, pmin, pmax, step=0.01):
        price_range = self.price_range(pmin, pmax, step=step)
        supply = np.zeros(len(price_range))
        for os in self.supply:
            supply += os.differential(pmin, pmax, step)
        
        demand = np.zeros(len(price_range))
        for od in self.demand:
            demand += od.differential(pmin, pmax, step)

        return np.cumsum(supply), np.cumsum(demand)
    
    def curves(self, pmin, pmax, step=0.01, fit_to_data=False):
        if fit_to_data:
            return self._curves_fit(step=step)
        else:
            return self._curves_not_fit(pmin, pmax, step=step)
        
    def divide_order_book(self, spot):
        x = self.signs * self.volumes * (spot - self.p0s)
        y = self.signs * self.volumes * (spot - self.prices - self.p0s)

        fully_rejected = SimpleOrderBook(self.orders[np.where(x < 0)])
        fully_accepted = SimpleOrderBook(self.orders[np.where(y > 0)])
        partially_accepted = SimpleOrderBook(
            self.orders[np.array(list(set(np.where(y <= 0)[0]).intersection(
                set(np.where(x >= 0)[0]))))])
        return fully_rejected, fully_accepted, partially_accepted

class LoadedOrderBook(SimpleOrderBook):
    """
    Same as SimpleOrderBook but everything is a tensor.
    Inout orders are regular list of orders that are converted into tensors:
    volumes, prices, p0s and signs.
    """
    def __init__(self, date_time, data_folder, volume_lines=False):
        self.date_time = date_time
        self.data_folder = data_folder
        self.volume_lines = volume_lines
        
        # Load from data_folder and disagregate
        datetime_str = datetime.datetime.strftime(date_time, "%Y-%m-%d_%Hh")
        supply_volumes = np.load(os.path.join(
            self.data_folder, f"{datetime_str}_supply_volumes.npy"))
        supply_prices = np.load(os.path.join(
            self.data_folder, f"{datetime_str}_supply_prices.npy"))
        demand_volumes = np.load(os.path.join(
            self.data_folder, f"{datetime_str}_demand_volumes.npy"))
        demand_prices = np.load(os.path.join(
            self.data_folder, f"{datetime_str}_demand_prices.npy"))

        # Revert the demand orders if not correct (NL and BE)
        if demand_prices[0] > demand_prices[-1]:
            demand_prices = demand_prices[::-1]
            demand_volumes = demand_volumes[::-1]        
        
        # Disagregate to form orders
        supply_orders = [StepOrder("Supply", supply_prices[0], supply_volumes[0])]
        demand_orders = [StepOrder("Demand", demand_prices[-1], demand_volumes[-1])]
        
        for volumes, prices, side, container, iteration, operation in zip(
                [supply_volumes, demand_volumes],
                [supply_prices, demand_prices],
                ["Supply", "Demand"],
                [supply_orders, demand_orders],
                [range(len(supply_volumes)-1),range(len(demand_volumes)-1, 0, -1)],
                [lambda x, y: x + y, lambda x, y: x - y]):
            
            reset_ptemp = False
            ptemp = prices[iteration[0]]
            for i in iteration:
                vi = volumes[i]
                pi = prices[i]
                vi1 = volumes[operation(i, 1)]
                pi1 = prices[operation(i, 1)]
                
                # If volumes are identical, store the last price!
                if (vi == vi1) and (reset_ptemp):
                    ptemp = pi
                    reset_ptemp = False
                if vi1 > vi:
                    if (not reset_ptemp) and (self.volume_lines):
                        reset_ptemp = True
                        print(f"Using price {ptemp} in place of price {pi}  at volume {vi}")
                        p_to_use = ptemp
                    else:
                        p_to_use = pi
                        
                    # If prices are identical, this is a step order
                    if p_to_use == pi1:
                        container += [StepOrder(side, p_to_use, vi1 - vi)]
                    else:
                        # Otherwise, this is a linear order
                        container += [LinearOrder(side, p_to_use, pi1, vi1 - vi)]

        SimpleOrderBook.__init__(self, supply_orders + demand_orders)
        
        
class TorchOrderBook(SimpleOrderBook):
    """
    Same as SimpleOrderBook but everything is a tensor.
    Inout orders are regular list of orders that are converted into tensors:
    volumes, prices, p0s and signs.
    """
    def __init__(self, orders, dtype=torch.float32, requires_grad=True):
        # If given a list of orders
        if ((type(orders).__name__ == "list")
            or ((type(orders).__name__ == "ndarray") and len(orders.shape) == 1)):
            SimpleOrderBook.__init__(self, orders)         
            volumes = torch.tensor([o.V * o.sign for o in self.orders],
                                   dtype=dtype, requires_grad=requires_grad)
            prices = torch.tensor([o.P for o in self.orders],
                                  dtype=dtype, requires_grad=requires_grad)
            p0s = torch.tensor([o.p0 for o in self.orders],
                               dtype=dtype, requires_grad=requires_grad)
            if requires_grad:
                volumes.retain_grad()
                prices.retain_grad()
                p0s.retain_grad()
        else:
            # If given a list of vectors
            volumes = orders[:, 0]
            prices = orders[:, 2]
            p0s = orders[:, 1]            

            f = lambda x: x.detach().numpy() if type(x).__name__ == "Tensor" else x
            directions = ["Supply" if v > 0 else "Demand" for v in f(volumes)]
            
            os = [LinearOrder(d, p0, p0 + p, abs(v)) if p != 0
                  else StepOrder(d, p0, abs(v))
                  for d, p0, p, v in zip(
                          directions, f(p0s), f(prices), f(volumes))]
            
            SimpleOrderBook.__init__(self, os)

        self.requires_grad = requires_grad
        self.dtype = dtype
        self.vs = volumes
        self.ps = prices
        self.pzeros = p0s

    @property
    def lin_orders(self):
        return np.array([o.is_lin for o in self.orders])

    @property
    def step_orders(self):
        return np.array([not o.is_lin for o in self.orders])
    
    @property
    def vlin(self):
        return self.vs[self.lin_orders]

    @property
    def plin(self):
        return self.ps[self.lin_orders]

    @property
    def p0lin(self):
        return self.pzeros[self.lin_orders]

    @property
    def vstep(self):
        return self.vs[self.step_orders]

    @property
    def pstep(self):
        return self.ps[self.step_orders]

    @property
    def p0step(self):
        return self.pzeros[self.step_orders]

    @property    
    def data(self):
        return torch.cat([
            self.vs.reshape(-1, 1),
            self.pzeros.reshape(-1, 1),
            self.ps.reshape(-1, 1)], axis=1).reshape(1, -1, 3)

    def accepted_volume(self, l):
        return np.sum([o.accepted_volume(l) for o in self.supply])
        
class BinaryOrderBook(SimpleOrderBook):
    """
    Order books with only 2 orders : 1 supply and 1 demand
    """
    def __init__(self, orders):       
        SimpleOrderBook.__init__(self, orders)
        
        if len(orders) != 2:
            raise InvalidInputError("BinaryOrderBook only accepts 2 orders.")

        if ((orders[0].is_demand and orders[1].is_demand)
            or (orders[0].is_supply and orders[1].is_supply)):
            raise InvalidInputError(
                "BinaryOrderBook should have 1 demand and 1 supply order.")

    @property
    def os(self):
        return self.supply[0]        
        
    @property
    def od(self):
        return self.demand[0]

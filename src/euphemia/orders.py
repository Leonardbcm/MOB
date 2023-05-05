import matplotlib.pyplot as plt, numpy as np


class InvalidInputError(Exception):
    def __init__(self, s):
        Exception.__init__(self, s)
        

class Order(object):
    """
    Base class for all orders. The only common behavior among all orders is the 
    direction : supply or demand
    """
    def __init__(self, direction):
        self.direction = direction
        
    @property
    def is_demand(self):
        return self.direction == "Demand"

    @property
    def is_supply(self):
        return not self.is_demand

    @property
    def sign(self):
        if self.is_demand:
            return -1
        else:
            return 1

    @staticmethod
    def dirac(x):
        if x == 0:
            return 1
        else:
            return 0

    @staticmethod
    def continuous_dirac(x, k):
        return Order.sigmoid(x, k) * (1 - Order.sigmoid(x, k))

    @staticmethod
    def sigmoid(x, k):
        return 1 / (1 + np.exp(- k * x))

    @staticmethod
    def heaviside(x):
        return np.heaviside(x, 1)
    
    def __eq__(self, other):
        return other.direction == self.direction
    

def simple_order(V, Po, P=0.0):
    if V < 0:
        direction = "Demand"
    if V > 0:
        direction = "Supply"
    if V == 0:
        raise ValueError("V is null!")
    
    # Create a step order
    if P == 0.0:
        return StepOrder(direction, Po, V)

    # Create a Linear order
    return LinearOrder(direction, Po, P, V)

    
class SimpleOrder(Order):
    """
    Step and Linear Orders
    """
    def __init__(self, direction):
        Order.__init__(self, direction)

    @property
    def V(self):
        return self.v        

    def price_range(self, step, pmin=None, pmax=None):
        if pmin is None:
            if self.is_demand:
                pmin = self.p0 + self.P
            else:
                pmin = self.p0
                
        if pmax is None:
            if self.is_demand:
                pmax = self.p0
            else:
                pmax = self.p0 + self.P
                
        return np.arange(pmin, pmax + step / 2, step)

    def dual_derivative(self, l, method="piecewise"):
        """
        Computes the dual derivative of this order using the given method
        """
        if method == "heaviside":
            return self.dual_derivative_heaviside(l)
        return self.dual_derivative_piecewise(l)        
        
    def dual_derivatives(self, l):
        return (self.dual_derivative(l, method="piecewise"),
                self.dual_derivative(l, method="heaviside"))

    def dual_derivative_generic_heaviside(self, l):
        """
        A generic formulation of the dual derivative for an Order.
        This will work on both step + linear orders.
        
        Implementation uses heaviside function
        """
        x = self.sign * self.V * (l - self.p0)
        y = self.sign * self.V * (l - self.p0 - self.P)
        
        Sx = Order.heaviside(x)
        Sy = Order.heaviside(y)
        
        return self.V*self.sign*Sy + x * (Sx - Sy) / (self.P + Order.dirac(self.P))

    def dual_derivative_generic_sigmoid(self, l, k=30):
        """
        A generic formulation of the dual derivative for an Order.
        This will work on both step + linear orders.
        
        Implementation approximates heaviside by a sigmoid function with factor k
        """        
        x = self.sign * self.V * (l - self.p0)
        y = self.sign * self.V * (l - self.p0 - self.P)
        z = - np.abs(self.P)

        Sx = Order.sigmoid(x, k)
        Sy = Order.sigmoid(y, k)
        dz = Order.continuous_dirac(self.P, k)

        return self.V*self.sign * Sy + x * (Sx - Sy) / (self.P + dz)
        
        
class StepOrder(SimpleOrder):
    """
    Defined by 1 price, 1 volume
    """
    def __init__(self, direction, p, v):
        SimpleOrder.__init__(self, direction)
        self.p = p
        self.v = v

    @property
    def is_lin(self):
        return False

    @property
    def p0(self):
        return self.p

    @property
    def P(self):
        return 0.0

    def is_full(self, l):
        return self.sign*self.V*(l - self.p0) >= 0

    def is_rejected(self, l):
        return not self.is_full(l)

    def accepted_volume(self, l):
        if self.is_full(l):
            return self.sign*self.V
        
        if self.is_rejected(l):
            return 0    

    def differential(self, pmin, pmax, step=0.01):
        """
        Volume variation against prices. Used for plotting the aggregated curves.
        """
        prices = self.price_range(step, pmin, pmax)
        volume_variations = np.zeros(len(prices))
        
        ind = int((self.p0 - pmin) / step)
        if self.is_demand:
            volume_variations[0] = self.V
            volume_variations[ind] = self.V * self.sign
        else:
            volume_variations[ind] = self.V        

        return volume_variations

    def dual_function(self, l):        
        if self.is_full(l):
            return self.sign*self.V*(l - self.p0)
        if self.is_rejected(l):
            return 0
        else:
            print(self, l)

    def dual_derivative_piecewise(self, l):
        """
        A litteral formulation of the dual derivative for a Step Order.
        """       
        # Fully accepted
        if self.is_full(l):
            return self.sign*self.V
        # Fully rejected
        if self.is_rejected(l):
            return 0
        else:
            print(self, l)

    def dual_derivative_heaviside(self, l):
        """
        A formulation of the dual derivative for a Step Order.
        Uses heaviside function in place of if statements
        """               
        x = self.sign * self.V * (l - self.p0)        
        return self.sign*self.V * Order.heaviside(x)

    def dual_derivative_sigmoid(self, l, k=30):
        """
        A formulation of the dual derivative for a Step Order.
        Approxmiates heaviside by a sigmoid
        """        
        x = self.sign * self.V * (l - self.p0)                
        return self.sign*self.V * Order.sigmoid(x, k)
    
    def __repr__(self):
        s = self.direction + "(" + str(self.v) + "MWh, " + str(self.p0) + "E)"
        return s

    
class LinearOrder(SimpleOrder):
    """
    Defined by 2 prices, 1 volume.
    The user can also specify a starting volume v0 for graphical solutions
    """
    def __init__(self, direction, p1, p2, v, v0=0):
        SimpleOrder.__init__(self, direction)
        if self.is_demand and p1 <= p2:
            raise InvalidInputError(
                "For a Demand order, p2 should be lower than p1.")

        if self.is_supply and p2 <= p1:
            raise InvalidInputError(
                "For a Supply order, p2 should be higher than p1.")

        if v <= 0:
            raise InvalidInputError(
                "V should be > 0!")        
        
        self.v = v
        self.v0 = v0
        self.v1 = v0 + v
        
        self.p1 = p1
        self.p2 = p2

    @property
    def is_lin(self):
        return True        

    @property
    def P(self):
        return self.p2 - self.p1

    @property
    def p0(self):
        return self.p1

    @property
    def V(self):
        return self.v

    @property
    def deltaV(self):
        return self.V / self.P

    def is_partial(self, l):
        return ((self.p0 <= l) and (l <= self.p0 + self.P)) or ((self.p0 >= l) and (l >= self.p0 + self.P))

    def is_full(self, l):
        return self.sign*self.V*(l - self.P - self.p0) > 0

    def is_rejected(self, l):
        return self.sign*self.V*(self.p0 - l) > 0

    def differential(self, pmin, pmax, step=0.01):
        """
        Volume variation against prices. Used for plotting the aggregated curves.
        """        
        prices = self.price_range(step, pmin, pmax)
        volume_variations = np.zeros(len(prices))

        pstart = self.p0 
        pstop = self.p0 + self.P        
        if self.is_demand:
            pstart, pstop = pstop, pstart
        
        indstart = int((pstart - pmin) / step)
        indstop = int((pstop - pmin) / step)        

        if not self.is_demand:
            volume_variations[indstart] = self.v0
            volume_variations[indstart+1:indstop+1] = self.deltaV * step
        else:            
            volume_variations[indstart+1:indstop+1] = self.deltaV * step
            volume_variations[0] = self.v0 + self.V
            
        return volume_variations

    def social_welfare(self, a):
        # Volumes are negative for demand orders
        V = self.V * self.sign 
        return - 0.5 * a * a * V * self.P - a * V * self.p0

    def energy_balance(self, a):
        # Volumes are negative for demand orders
        V = self.V * self.sign 
        return a * V

    def dual_function(self, l):        
        if self.is_partial(l):            
            return (l - self.p0)*(l - self.p0)*self.V*self.sign/(2*self.P)
        
        if self.is_full(l):
            return self.sign*self.V*(l - self.P/2 - self.p0)
        
        if self.is_rejected(l):
            return 0
        else:
            print(self, l)

    def accepted_volume(self, l):        
        if self.is_partial(l):            
            return (l - self.p0)*self.V*self.sign/self.P
        
        if self.is_full(l):
            return self.sign*self.V
        
        if self.is_rejected(l):
            return 0
            
    def dual_derivative_piecewise(self, l):
        """
        Computes the dual derivative for a Linear Order using if statements
        """               
        if self.is_partial(l):
            return (l - self.p0) * self.V * self.sign / self.P
        
        # Fully accepted
        if self.is_full(l):
            return self.sign*self.V

        # Fully rejected
        if self.is_rejected(l):
            return 0
        else:
            print(self, l)
    
    def dual_derivative_heaviside(self, l):
        """
        Computes the dual derivative for a Linear Order using heaviside
        """               
        x = self.sign * self.V * (l - self.p0)
        y = self.sign * self.V * (l - self.p0 - self.P)
        
        return (x * Order.heaviside(x) - y * Order.heaviside(y)) / self.P    
    
    def dual_derivative_sigmoid(self, l, k=30):
        """
        Computes the dual derivative for a Linear Order by approximating the 
        heaviside function with a sigmoid with factor k.
        """         
        x = self.sign * self.V * (l - self.p0)
        y = self.sign * self.V * (l - self.p0 - self.P)
        
        return (x * Order.sigmoid(x, k) - y * Order.sigmoid(y, k)) / self.P      

    def __eq__(self, other):        
        return (Order.__eq__(self, other) and (self.p1 == other.p1)
                and (self.p2 == other.p2) and (self.v == other.v)
                and (self.v0 == other.v0))

    def __repr__(self):
        s = self.direction + "("
        if self.v0 != 0:
            s += str(self.v0) + "->"
        s += str(self.v1)+"MWh, "+str(self.p0)+"E->" + str(self.p0 + self.P) + "E)"
        return s

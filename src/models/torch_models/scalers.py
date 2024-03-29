import torch
from torch import nn

class TorchMinMaxScaler(nn.Module):
    """
    A MinMaxScaler that maps data X to the interval [a, b]. Apply the following:
    Xstd = (X - min(X)) / (max(X) - min(X))
    Xout = Xtsd  * (a - b) + b

    for the last dimmension of the input, with min(X) = min([Po, P+Po]).
    """
    def __init__(self, a=0, b=1, dtype=torch.float32):
        nn.Module.__init__(self)
        self.a = a
        self.b = b
        self.dtype = dtype

    def forward(self, X):
        Xmins = torch.min(X, axis=1, keepdim=True)[0]
        Xmaxs = torch.max(X, axis=1, keepdim=True)[0]

        self.Xmins = Xmins
        self.Xmaxs = Xmaxs        
        
        Xstd = (X - Xmins) / (Xmaxs - Xmins)
        Xscaled = Xstd * (self.a - self.b) + self.b        

        return Xscaled

    def __str__(self):
        a_round = round(self.a, ndigits=2)
        b_round = round(self.b, ndigits=2)        
        return f"TorchMinMaxScaler({a_round}, {b_round})"

    def __repr__(self):
        return self.__str__()
        
        
class TorchMinAbsScaler(nn.Module):
    """
    A MinAbs Scaler. Mutliply the data by a factor A so that the minimal absolute
    value is m.
    A = m / min(|X|)
    Xscaled = AX
    """
    def __init__(self, m=0.1, dtype=torch.float32):
        nn.Module.__init__(self)        
        self.m = m
        self.dtype = dtype

    def forward(self, x):
        xabs = torch.abs(x)
        minabs = torch.min(xabs, axis=1, keepdim=True)[0]
        xout = x * self.m / minabs
        return xout

    
class TorchCliper(nn.Module):
    """
    Clip data in the followinf interval [a, b]
    """
    def __init__(self, a, b, min_frac=1.0, max_frac=1.0, dtype=torch.float32):
        nn.Module.__init__(self)
        self.a = a * min_frac
        self.b = b * max_frac
        self.dtype = dtype

    def forward(self, x):
        xout = torch.clip(x, min=self.a, max=self.b)
        return xout

    def __str__(self):
        a_round = round(self.a, ndigits=2)
        b_round = round(self.b, ndigits=2)        
        return f"TorchCliper({a_round}, {b_round})"

    def __repr__(self):
        return self.__str__()    

    
class AbsCliper(nn.Module):
    """
    Clip data so that the min absolute value of the data is higher
    that a threshold m
    Use the sigmoid trick to compute the sign function
    """
    def __init__(self, m, k=30, dtype=torch.float32):
        nn.Module.__init__(self)        
        self.m = m
        self.k = k
        self.dtype = dtype

    def forward(self, x):
        signs = 2.0 * torch.sigmoid(self.k * x) - 1.0        
        cliped = torch.clip(torch.abs(x), min=self.m)        
        return signs * cliped

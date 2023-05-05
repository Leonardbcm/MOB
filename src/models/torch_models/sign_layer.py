import numpy as np, torch
from torch import nn

class FixedSignLayer(nn.Module):
    """
    A layer that coerce the signs of the input.
    Rules for the signs are :
    
    The first OBs/2 orders are supply orders, the last
    OBs/2 orders are demand orders
    """
    def __init__(self, OBs, dtype=torch.float32):
        nn.Module.__init__(self)
        self.dtype = dtype
        self.OBs = OBs
        self.middle = int(self.OBs / 2)

        self.mask = torch.ones(self.OBs)
        self.mask[self.middle:] = -1

    def forward(self, P, V):        
        # Apply signs
        V = torch.abs(V) * self.mask
        P = torch.abs(P) * self.mask
        return P, V

    def __str__(self):
        return f"FixedSignLayer({self.mask})"

    def __repr__(self):
        return self.__str__()    

class SignLayer(nn.Module):
    """
    A layer that coerce the signs of the input.
    Rules for the signs are :
    
    If P > 0, V > 0
    If P < 0, V < 0
    If P = 0, V does not changes
    """
    def __init__(self, k, scale, dtype=torch.float32):
        nn.Module.__init__(self)
        self.dtype = dtype
        self.k = k
        self.scale = scale

    def forward_clip_sign(self, P, V):
        # a) Compute sigmoids
        Sp = torch.sigmoid(self.k * P)
        Sv = torch.sigmoid(self.k * V)

        # b) Compute signs
        self.signs_P = 2.0 * Sp - 1.0
        signs_V = 2.0 * Sv - 1.0
    
        # c) Correct the signs of V when P = 0
        dirac_P = 4 * Sp * (1 - Sp)
        self.signs_V = self.signs_P + dirac_P * signs_V
        
        # d) Apply signs
        V = torch.abs(V) * self.signs_V
        P = torch.abs(P) * self.signs_P

        return P, V

    def forward_(self, P, V):
        # a) Compute sigmoids
        Sp = torch.sigmoid(self.k * P)

        # b) Compute signs
        self.signs_P = 2.0 * Sp - 1.0 + 4 * Sp * (1 - Sp)   

        # c) Apply signs
        V = torch.abs(V) * self.signs_P
        P = torch.abs(P) * self.signs_P

        return P, V

    def forward(self, P, V):
        if self.scale == "Clip-Sign":
            return self.forward_clip_sign(P, V)
        else:
            return self.forward_(P, V)

    def __str__(self):
        return f"SignLayer({self.scale}, k={self.k})"

    def __repr__(self):
        return self.__str__()    

if __name__ == "__main__":
    def signs(P, V):
        Sp = torch.sigmoid(30 * P)
        Sv = torch.sigmoid(30 * V)
        
        Ps = 2 * Sp - 1
        Vs = 2 * Sv - 1
        Pd = 4 * Sp * (1 - Sp)
        
        return Ps + Pd * Vs

    P = torch.tensor(np.arange(-1, 1, 0.001))
    V = torch.tensor(np.arange(-1, 1, 0.001))
    res = np.zeros((len(P), len(V)))

    for x, p in enumerate(P):
        res[x, :] = signs(p, V)
        
    plt.imshow(res.transpose(), origin='lower', cmap="seismic")
    plt.show()
    

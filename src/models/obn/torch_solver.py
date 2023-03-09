import torch
from torch import nn

class Solver(nn.Module):
    """
    Abstract class for all solver : Single And Batch
    """
    def __init__(self, lb, ub, k=30, mV=0.1, retain_grad=False,
                 dtype=torch.float32, check_data=True):
        nn.Module.__init__(self)

        self.ub_ = ub
        self.lb_ = lb
        self.k = k
        self.mV = mV
        self.retain_grad = retain_grad
        self.check_data_ = check_data
        self.dtype = dtype

    def check_data(self, data, name):
        """
        Check if the input data is in the right bounds (prices should be in a 
        certain interval)
        Also, the minimal absolute volume should be at least a certain value.
        """
        if name == "V, P":
            v1 = torch.sign(data[0])
            v2 = torch.sign(data[2])
            if (v1 != v2).any():
                estring = f"Invalid input in the data : the sign of the volumes differs from the sign of the prices sign(V) != sign(P)!"
                if self.check_data_:
                    raise Exception(estring)
            
        if len(data.shape) > 1:
            func = lambda x, f: f(x, axis=1)[0]            
        else:
            func = lambda x, f: f(x)

        if name in ("Po", "Po + P"):
            v = func(data, torch.min)
            if (v < self.pmin).any():
                values = v[v < self.pmin]
                estring = f"Invalid input in the data : some elements of {name} are lower than pmin! \n {values.numpy()} < {self.pmin}"
                if self.check_data_:
                    raise Exception(estring)
                
            v = func(data, torch.max)            
            if (v > self.pmax).any():
                values = v[v > self.pmax]
                estring = f"Invalid input in the data : some elements of {name} are higher than pmax! \n {values.numpy()} > {self.pmax}"
                if self.check_data_:
                    raise Exception(estring)
        if name == "V":
            v = func(torch.abs(data), torch.min)
            if (v < self.mV).any():
                values = v[v < self.mV]
                estring = f"Invalid input in the data : some elements of {name} are in the interval ]-mV, mV[! \n |{values.numpy()}| < {self.mV}"
                if self.check_data_:
                    raise Exception(estring)        
                

class DichotomicSolver(Solver):
    """
    An abstract dichotomic solver in torch. The forward method consists in solving
    the dichotomic search of the 0 of a given dual_derivative function. 
    
    The gradient of the solution with respect to the input is handled by torch.
    To use this class, create a sub-class that inherit the forward method and 
    implements the dual_derivative method.
    """
    def __init__(self, lb, ub, step, k=30, mV=0.1, retain_grad=False,
                 dtype=torch.float32, check_data=True):
        Solver.__init__(self, lb, ub, k=k, mV=mV, retain_grad=retain_grad,
                        dtype=dtype, check_data=check_data)
        self.step = step        
        self.init_parameters()        
        
    def init_parameters(self):
        # Counts the number of steps
        self.steps = 0
        
        # Create the lower and upper bounds
        self.lb = torch.tensor(
            self.lb_,dtype=self.dtype, requires_grad=self.retain_grad)
        self.ub = torch.tensor(
            self.ub_,dtype=self.dtype, requires_grad=self.retain_grad)

        # Intialize gradient containers
        if self.retain_grad:
            self.lbs = []
            self.ubs = []
            self.ms = []        
            self.HDs = []
            self.DMs = []        

    def forward(self, x):
        """ 
        The shape of x should be bs x (s) with s the expected shape of the
        dual_derivative function.

        This solve the problem for all data points of the first dimmension.
        this function uses for loops and won't be efficient!

        The output shape will be bs x 1
        """
        out = torch.concatenate([self.solve(x[i]) for i in range(x.shape[0])])
        return out.reshape(-1, 1)
            
    def solve(self, x):
        """ 
        The shape of x should be the shape expected by the dual derivative function.
        """ 
        self.init_parameters()        
        
        found = False
        while (not found) and (self.ub - self.lb > 2 * self.step):
            m = (self.lb + self.ub) / 2
            Dm = self.dual_derivative(x, m)
            found = torch.abs(Dm) < self.step

            HDM = torch.sigmoid(self.k * Dm)
            
            self.lb = m - HDM * (m - self.lb)            
            self.ub = self.ub - HDM * (self.ub - m)

            if self.retain_grad:
                m.requires_grad_(True)
                m.retain_grad()
                
                HDM.requires_grad_(True)                
                HDM.retain_grad()

                Dm.requires_grad_(True)                
                Dm.retain_grad()
                
                self.lb.retain_grad()
                self.ub.retain_grad()

                self.ms += [m]
                self.HDs += [HDM]
                self.DMs += [Dm]            
                self.lbs += [self.lb]
                self.ubs += [self.ub]
            
            self.steps += 1

        return m.reshape(-1)

    def dual_derivative(self, x, l):
        raise(NotImplementedError("The dual_derivative of the DichotomicSolver abstract class is not implemented!"))
    

class PFASolver(DichotomicSolver):
    """
    Solves the PFA using a dichotomic search. The dual_derivative function is the
    derivative of the dual problem of euphemia whose solution is the day ahead price
    """
    def __init__(self, pmin=-500, pmax=3000, step=0.01, k=30, mV=0.1,
                 retain_grad=False, check_data=True):
        DichotomicSolver.__init__(
            self, pmin, pmax, step, k, mV=mV, retain_grad=retain_grad,
            check_data=check_data)
        self.pmin = pmin
        self.pmax = pmax        

    def separate_data(self, data):
        """
        Returns V, Po, P given a data object of shape OBs X 3
        Returned object are of the shape OBs
        """
        V = data[:, 0]
        Po = data[:, 1]
        P = data[:, 2]

        self.check_data(V.detach(), "V")
        self.check_data(Po.detach(), "Po")
        self.check_data(Po.detach() + P.detach(), "Po + P")
        self.check_data((P.detach(), V.detach()), "V, P")
        
        return V, Po, P        

    def dual_derivatives(self, data, l):
        """
        Dual derivative of all orders

        The shape of tensor data is excpected to be : OBs x 3
        With bs the batch size, OBs the order book size.
        In the last dimmension, at index:
        0 are the volumes V
        1 are the initial prices Po
        2 are the price ranges P

        The output will be a tensor with 1 element!
        """
        v, p0, p = self.separate_data(data)
        
        x = v * (l - p0)
        y = v * (l - p0 - p)        
        z = - torch.abs(p)
        
        Sx = torch.sigmoid(self.k * x)
        Sy = torch.sigmoid(self.k * y)
        Sz = torch.sigmoid(self.k * z)        
        
        dz = Sz * (1 - Sz)
        return v * Sy + x * (Sx - Sy) / (p + dz)

    def dual_derivative(self, x, l):
        return torch.sum(self.dual_derivatives(x, l))


class BatchDichotomicSolver(Solver):
    """
    An abstract batch  dichotomic solver in torch. 
    The forward method consists in solving the dichotomic search of the 0 of 
    a given dual_derivative function for an entiere batch of data.

    During the search, the while loop is replaced by a for loop with a fixed
    number of iterations niter. We recommand setting this number to 
    2**niter = card(S) with S the solution space. This way it will be enough
    iterations to explore the search space.
    
    Normally, keep iterating after finding the solution does not matters because 
    when DM = 0, HDM = 0.5 and lb/ub just iterates over themselves.
    
    The gradient of the solution with respect to the input is handled by torch.
    To use this class, create a sub-class that inherit the forward method and 
    implements the dual_derivative method.
    """
    def __init__(self, niter, lb, ub, k=30, mV=0.1, retain_grad=False,
                 dtype=torch.float32, check_data=False):
        Solver.__init__(self, lb, ub, k=k, mV=mV, retain_grad=retain_grad,
                        dtype=dtype, check_data=check_data)
        self.niter = niter        

    def init_parameters(self, bs):
        # Counts the number of steps
        self.steps = 0
        
        # Create the lower and upper bounds
        self.LB = self.lb_ * torch.ones(
            (bs, 1), dtype=self.dtype, requires_grad=self.retain_grad)
        
        self.UB = self.ub_ * torch.ones(
            (bs, 1), dtype=self.dtype, requires_grad=self.retain_grad)
        
        # Intialize gradient containers
        if self.retain_grad:
            self.lbs = []
            self.ubs = []
            self.ms = []        
            self.HDs = []
            self.DMs = []        

    def forward(self, x):
        """ 
        The shape of x should be bs x (s) with s the expected shape of the
        dual_derivative function.

        This solve the problem for all data points of the first dimmension in 
        a batch manner.

        The output shape will be bs x 1
        """
        out = self.solve(x)
        return out.reshape(-1, 1)
            
    def solve(self, x):
        """ 
        The shape of x should be the shape expected by the dual derivative function.
        """
        bs = x.shape[0]
        self.init_parameters(bs)
        
        for i in range(self.niter):
            m = (self.LB + self.UB) / 2
            Dm = self.dual_derivative(x, m)
            HDM = torch.sigmoid(self.k * Dm)
            
            self.LB = m - HDM * (m - self.LB)
            self.UB = self.UB - HDM * (self.UB - m)

            if self.retain_grad:
                m.requires_grad_(True)
                m.retain_grad()
                
                HDM.requires_grad_(True)                
                HDM.retain_grad()

                Dm.requires_grad_(True)                
                Dm.retain_grad()
                
                self.lb.retain_grad()
                self.ub.retain_grad()

                self.ms += [m]
                self.HDs += [HDM]
                self.DMs += [Dm]            
                self.lbs += [self.lb]
                self.ubs += [self.ub]
            
            self.steps += 1

        return m

    def dual_derivative(self, x, l):
        raise(NotImplementedError("The dual_derivative of the DichotomicSolver abstract class is not implemented!"))
    

class BatchPFASolver(BatchDichotomicSolver):
    """
    Solves the PFA using a dichotomic search over an entiere batch of data. 
    The dual_derivative function is the derivative of the dual problem
    of euphemia whose solution is the day ahead price.

    Problems are solved indenpendantly but at the same time.
    """
    def __init__(self, niter=20, pmin=-500, pmax=3000, k=30, mV=0.1,
                 retain_grad=False, check_data=False):
        BatchDichotomicSolver.__init__(self, niter, pmin,pmax,k,mV,
                                       retain_grad,check_data=check_data)
        self.pmin = pmin
        self.pmax = pmax
            
    def separate_data(self, data):
        """
        Returns V, Po, P given a data object of shape bs x OBs X 3
        Returned tensors are of shape bs x OBs

        Also check the input extremas. Po and P + Po should be in the range
        [pmin, pmax]
        """
        V = data[:, :, 0]
        Po = data[:, :, 1]
        P = data[:, :, 2]

        # Check the extremas of the inputs
        self.check_data(V.detach(), "V")
        self.check_data(Po.detach(), "Po")
        self.check_data(Po.detach() + P.detach(), "Po + P")        
        
        return V, Po, P        

    def dual_derivatives(self, data, L):
        """
        Dual derivative of all orders

        The shape of tensor data is excpected to be : bs x OBs x 3
        With bs the batch size, OBs the order book size.
        In the last dimmension, at index:
        0 are the volumes V
        1 are the initial prices Po
        2 are the price ranges P

        Input l must be a tensor with shape bs x 1
        
        The output will be a tensor with bs elements.
        """
        V, Po, P = self.separate_data(data)
        
        # Substract along axis=1
        lminusPo = L - Po
        lminusPoP = lminusPo - P

        # Multiply element-wise along axis=1
        X = V * lminusPo
        Y = V * lminusPoP
        Z = - torch.abs(P)

        # Apply sigmoid element-wise along all dims
        SX = torch.sigmoid(self.k * X)
        SY = torch.sigmoid(self.k * Y)
        SZ = torch.sigmoid(self.k * Z)        

        # Multiplication elementwise
        DZ = SZ * (1 - SZ)

        # Everything is applied elementwise
        return V * SY + X * (SX - SY) / (P + DZ)

    def dual_derivative(self, X, L):
        return torch.sum(self.dual_derivatives(X, L), axis=1, keepdim=True)
        

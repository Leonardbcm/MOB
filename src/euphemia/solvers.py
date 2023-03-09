import torch, numpy as np, pandas
from src.euphemia.orders import *
from src.euphemia.order_books import *

class IncompatibleOrderBookError(Exception):
    def __init__(self, s):
        Exception.__init__(self, s)

        
class Solver(object):
    """
    Price Fixing Algorithms : Solves the Dual problem for a given Order Book
    """
    def __init__(self, order_book):
        self.order_book = order_book


class BinarySolution(Solver):
    """
    Solution for binary Linear Binary Order Books
    This solution does not call the social welfare function of individual orders
    but directly use analytical solutions tailored for binary OB.
    """
    def __init__(self, order_book):
        self.order_book = order_book
        
        if not type(self.order_book).__name__ == "BinaryOrderBook":
            raise IncompatibleOrderBookError(
                "Graphic Solution can only be found on binary Order Books.")
        
        if not (type(self.order_book.os).__name__ == 'LinearOrder'
                and (type(self.order_book.od).__name__ == 'LinearOrder')):
            raise IncompatibleOrderBookError(
                "Graphic Solution can only be found with Linear orders.")
        
    @property
    def od(self):
        return self.order_book.od

    @property
    def os(self):
        return self.order_book.os

    def solve(self):
        od = self.order_book.od
        os = self.order_book.os
        
        v1 = od.V * os.V * (os.p0 - od.p0)
        v2 = os.V * od.P * od.v0
        v3 = - od.V * os.P * os.v0
        v4 = os.V * od.P - od.V * os.P
        vstar = (v1 + v2 + v3) / v4

        p1 = od.P * os.P * (od.v0 - os.v0)
        p2 = os.V * od.P * os.p0
        p3 = - od.V * os.P * od.p0
        p4 = os.V * od.P - od.V * os.P
        pstar = (p1 + p2 + p3) / p4
        return vstar, pstar    

        
class BinaryGraphicalSolution(BinarySolution):
    """
    Fix Prices of a binary Order Book using the graphical solution.
    """
    def __init__(self, order_book):
        BinarySolution.__init__(self, order_book)

        
class BinaryAnalyticSolution(BinarySolution):
    """
    Fix Prices of a binary Order Book using the analytic solution.
    """
    def __init__(self, order_book):
        BinarySolution.__init__(self, order_book)   
        
    def social_welfare(self, a_s, a_d):
        od = self.od
        os = self.os
        
        demand_surplus  = 0.5 * a_d * a_d * od.V * od.P + a_d * od.V * od.p0
        supply_surplus = - 0.5 * a_s * a_s * os.V * os.P - a_s * os.V * os.p0
        return demand_surplus + supply_surplus

    def energy_balance(self, a_s, a_d):
        od = self.od
        os = self.os
        return os.v0 + a_s * os.V - od.v0 - a_d * od.V

    def dual_function(self, l):
        od = self.od
        os = self.os
        quad = l * l * (os.V/os.P - od.V/od.P) / 2
        lin = l * (os.v0 - od.v0 + od.V*od.p0 /od.P - os.V*os.p0 / os.P)
        fixed = - 0.5*(od.V*od.p0*od.p0)/(od.P) + 0.5*(os.V*os.p0*os.p0)/(os.P)
        return quad + lin + fixed

    def lagrangian(self, a_s, a_d, l):
        od = self.od
        os = self.os
        t11 = 0.5 * a_d * a_d * od.V * od.P
        t12 =  a_d * od.V * od.p1
        t21 = - 0.5 * a_s * a_s * os.V * os.P
        t22 = - a_s * os.V * os.p1
        t3 = l * (os.v0 - od.v0 + a_s * os.V - a_d * od.V)
        return t11 + t12 + t21 + t22 + t3

    def compute_acceptance(self, l):
        os = self.order_book.os
        od = self.order_book.od
        a_s = (l  - os.p0) / os.P
        a_d = (l  - od.p0) / od.P
        return np.array([a_s, a_d])
    
        
class ComputationalSolution(Solver):
    """
    Fix Prices of a binary Order Book using the analytic solution.
    This function calls the orders social welfare functions.
    """
    def __init__(self, order_book, pmin=-500, pmax=3500, step=0.01):
        Solver.__init__(self, order_book)
        
        if pmax <= pmin:
            raise InvalidInputError("pmax should be higher than pmin.")
        if pmax - pmin <= step:
            raise InvalidInputError(
                "step should be lower than the price interval.")
        
        self.pmin = pmin
        self.pmax = pmax
        self.step = step        

    @property
    def price_range(self):
        return np.arange(self.Pmin, self.Pmax, self.step)

    def solve(self):
        raise NotImplementedError(
            "Please use the subclasses of ComputationalSolution")
    

class MinDual(ComputationalSolution):
    """
    Fix Prices of a binary Order Book using the analytic solution.
    This function calls the orders social welfare functions.
    """
    def __init__(self, order_book, pmin=-500, pmax=3500, step=0.01):
        ComputationalSolution.__init__(self, order_book)    

    def dual_function(self, l):
        dualf = np.sum([o.dual_function(l) for o in self.order_book.orders])
        return dualf

    def dual_derivative_lambda(self, l, method="piecewise", k=30):
        if "sigmoid" in method:
            derf = np.sum([o.dual_derivative_sigmoid(l, k=k)
                           for o in self.order_book.orders])                     
        else:
            derf = np.sum([getattr(o, f"dual_derivative_{method}")(l)
                           for o in self.order_book.orders])
                        
        return derf
        
    def solve(self, method="dual_function", k=30):
        if method == "real_price":
            return self.get_real_price()
        
        func = self.dual_derivative_lambda
        arg = method.split("dual_derivative_")[1]
        
        # Create the lower and upper bounds
        lb = self.pmin
        ub = self.pmax
        
        found = False
        while (not found) and (ub - lb > 2 * 0.01):
            m = (lb + ub) / 2
            if "sigmoid" in arg:
                value_at_m = func(m, method=arg, k=k)
            else:
                value_at_m = func(m, method=arg)
                
            found = np.abs(value_at_m) < 0.0001            
            if "heaviside" in arg:
                HDM = np.heaviside(value_at_m, 1)
            if arg == "piecewise":
                HDM = 0 
                if value_at_m >= 0:
                    HDM = 1
            if "sigmoid" in arg:
                def sigmoid(x, k=30):
                    return 1 / (1 + np.exp(- k * x))
                HDM = sigmoid(value_at_m, k=k)
            
            lb = m - HDM * (m - lb)   
            ub = ub - HDM * (ub - m)
            
        return m

    def get_real_price(self):
        folder = os.path.split(self.order_book.data_folder)[0]
        filename = "interpolated_price.csv"
        results = pandas.read_csv(os.path.join(folder, filename))
        string_date = datetime.datetime.strftime(
            self.order_book.date_time, "%Y-%m-%dT%H:%M:00.0")
        real_ = results.loc[results.period_start_time == string_date, "price_mean"]
        return real_.values[0]

    def solve_all_methods(self, ret_dict=False, ks=[1, 10, 100, 1000]):
        res = []
        methods = ["real_price",
                   "dual_derivative_piecewise",
                   "dual_derivative_heaviside",
                   "dual_derivative_generic_heaviside"]
        return_dict = {}
        for method in methods:
            result = self.solve(method=method)
            res += [result]
            return_dict[method] = result

        for k in ks:
            result = self.solve(method="dual_derivative_sigmoid", k=k)
            method = f"dual_derivative_sigmoid_{k}"
            methods += [method]
            res += [result]
            return_dict[method] = result
        
        for k in ks:
            result = self.solve(method="dual_derivative_generic_sigmoid", k=k)
            method = f"dual_derivative_generic_sigmoid_{k}"
            methods += [method]
            res += [result]
            return_dict[method] = result                     
            
        if ret_dict:
            return return_dict

        return res, methods

        
class OnlyPartial(ComputationalSolution):
    """
    Fix Prices of a binary Order Book using the analytic solution.
    This function calls the orders social welfare functions.
    """
    def __init__(self, order_book, pmin=-500, pmax=3500, step=0.01):
        ComputationalSolution.__init__(self, order_book)

    def solve(self):
        n = sum([o.sign*o.V*o.p0/o.P for o in self.order_book.orders])
        d = sum([o.sign*o.V/o.P for o in self.order_book.orders])
        l = n/d

        ais = np.array([(l - o.p0)/o.P for o in self.order_book.orders])
        vstar = sum([ai*o.V for (o, ai) in zip(self.order_book.orders, ais)]) / 2
        return vstar, l

    
class TorchSolution(ComputationalSolution):
    """
    Solving the problem with torch tensors. The output solution should be
    differentiable with respect to the input Order Books.

    Data conversion to tensor is made on the fly, this is done for checking
    If the torch methods are functionnal (able to solve the right price).
    """
    def __init__(self, order_book, pmin=-500, pmax=3000, step=0.01,
                 k=30, epsilon=0.0001):
        ComputationalSolution.__init__(self, order_book)        
        if not type(self.order_book).__name__ == "TorchOrderBook":
            raise IncompatibleOrderBookError(
                "Torch Solution only works with TorchOrderBooks")

        # k controls the approximation of the heaviside function by a sigmoid
        self.k = k

        # Epsilon is added to the denominators. also used to stop search for lambda
        self.epsilon = epsilon

        self.init_solver()

    def init_solver(self):
        # Store the number of steps during dichotomy
        self.steps = 0
        
        self.lbs = []
        self.ubs = []
        self.ms = []        
        self.HDs = []
        self.DMs = []
        
    @property
    def v(self):
        return self.order_book.vs

    @property
    def p(self):
        return self.order_book.ps

    @property
    def p0(self):
        return self.order_book.pzeros

    @property
    def V(self):
        return self.order_book.vs.detach().numpy()

    @property
    def P(self):
        return self.order_book.ps.detach().numpy()

    @property
    def P0(self):
        return self.order_book.pzeros.detach().numpy()
    
    @property
    def vlin(self):
        return self.order_book.vlin

    @property
    def plin(self):
        return self.order_book.plin

    @property
    def p0lin(self):
        return self.order_book.p0lin

    @property
    def vstep(self):
        return self.order_book.vstep

    @property
    def pstep(self):
        return self.order_book.pstep

    @property
    def p0step(self):
        return self.order_book.p0step

    def dual_derivatives_lambda_separated(self, l):
        x = self.vlin * (l - self.p0lin)
        y = self.vlin * (l - self.p0lin - self.plin)
        Sx = torch.nn.Sigmoid()(self.k * x)
        Sy = torch.nn.Sigmoid()(self.k * y)
        linear_dual = (x * Sx - y * Sy) / self.plin

        xstep = self.vstep * (l - self.p0step)
        Sxstep = torch.nn.Sigmoid()(self.k * xstep)        
        step_dual = self.vstep * Sxstep
        
        return torch.concat([linear_dual, step_dual])

    def dual_derivatives_lambda(self, l, epsilon=0.0):
        x = self.v * (l - self.p0)
        y = self.v * (l - self.p0 - self.p)        
        z = epsilon - torch.abs(self.p)
        
        Sx = torch.nn.Sigmoid()(self.k * x)
        Sy = torch.nn.Sigmoid()(self.k * y)
        Sz = torch.nn.Sigmoid()(self.k * z)        
        
        dz = Sz * (1 - Sz)
        return self.v * Sy + x * (Sx - Sy) / (self.p + dz)

    def dual_derivatives_lambda_protected(self, l, epsilon=0.0):
        x = self.v * (l - self.p0)
        y = self.v * (l - self.p0 - self.p)        
        
        Sx = torch.nn.Sigmoid()(self.k * x)
        Sy = torch.nn.Sigmoid()(self.k * y)
        
        # Replaces nan (0/0) by 0
        div = torch.nan_to_num((Sx - Sy) / self.p)
        
        return self.v * Sy + x * div    
        
    def dual_derivative_lambda(self, l):
        return torch.sum(self.dual_derivatives_lambda(l))

    def dual_derivative_lambda_separated(self, l):
        return torch.sum(self.dual_derivatives_lambda_separated(l))

    def dual_derivative_lambda_protected(self, l):
        return torch.sum(self.dual_derivatives_lambda_protected(l))    

    def solve_all_methods(self, ret_dict=False, ks=[1, 10, 100, 1000]):
        res = []
        return_dict = {}

        for k in ks:
            self.k=k
            result = self.solve().detach().item()
            res += [result]
            return_dict[f"sigmoid_{k}"] = result

        for k in ks:
            self.k=k            
            result = self.solve(protected=True).detach().item()
            res += [result]
            return_dict[f"protected_{k}"] = result            

        for k in ks:
            self.k=k            
            result = self.solve(generic=True).detach().item()
            res += [result]
            return_dict[f"separated_{k}"] = result            
            
        if ret_dict:
            return return_dict

        return res, list(return_dict.keys())
    
    def solve(self, generic=False, protected=False, niter=0):
        self.init_solver()        
        
        # Solve using a dichotomy search. If statements are replaced by heaviside
        # functions, approximated by sigmoid

        # Create the lower and upper bounds
        lb = torch.tensor(self.pmin, dtype=float)
        ub = torch.tensor(self.pmax, dtype=float)
        
        found = False
        while ((not found) and (ub - lb > 2 * self.step)) or (self.steps < niter):
            m = (lb + ub) / 2
            if (not generic) and (not protected):
                value_at_m = self.dual_derivative_lambda(m)
            elif generic:
                value_at_m = self.dual_derivative_lambda_generic(m)
            elif protected:
                value_at_m = self.dual_derivative_lambda_protected(m)
                
            found = torch.abs(value_at_m) < self.epsilon
            HDM = torch.sigmoid(self.k * value_at_m)
            
            lb = m - HDM * (m - lb)            
            ub = ub - HDM * (ub - m)

            m.requires_grad_(True)
            m.retain_grad()
            HDM.retain_grad()
            value_at_m.retain_grad()            
            lb.retain_grad()
            ub.retain_grad()    

            self.ms += [m]
            self.HDs += [HDM]
            self.DMs += [value_at_m]            
            self.lbs += [lb]
            self.ubs += [ub]
            
            self.steps += 1

        return m

    def get_gradients(self):
        grad_ubs = np.array([ub.grad for ub in self.ubs])
        grad_lbs = np.array([lb.grad for lb in self.lbs])
        grad_ms = np.array([m.grad for m in self.ms])
        grad_HDs = np.array([HD.grad for HD in self.HDs])
        grad_DMs = np.array([DM.grad for DM in self.DMs])
        return grad_ubs, grad_lbs, grad_ms, grad_HDs, grad_DMs

    def detach(self):
        self.order_book.vs = self.order_book.vs.detach()
        self.order_book.ps = self.order_book.ps.detach()
        self.order_book.pzeros = self.order_book.pzeros.detach()

    def compute_ratios(self, l):
        x = self.V * (l - self.P0)
        y = self.V * (l - self.P0 - self.P)        
        
        Sx = 1 - np.heaviside(x, 1)
        Sy = np.heaviside(y, 1)
        
        fr = np.sum(Sx)
        fa = np.sum(Sy)
        pa = self.order_book.n - fr - fa
        return fr, fa, pa
        

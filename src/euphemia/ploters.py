from src.euphemia.orders import *
from src.euphemia.order_books import *
from src.euphemia.solvers import *
import matplotlib.pyplot as plt, numpy as np, math, copy

def get_ploter(order_book, solver=None, results=(None, None)):
    if solver is None:
        return SimplePloter(order_book, None, None)
    if type(solver).__name__ == 'BinaryGraphicalSolution':
        return BinaryGraphicalPloter(order_book, solver, results)
    if type(solver).__name__ == 'BinaryAnalyticSolution':
        return BinaryAnalyticPloter(order_book, solver, results)
    return SimplePloter(order_book, solver, results)
    

class Ploter(object):
    def __init__(self, order_book, solver, results):
        self.order_book = order_book
        self.solver = solver
        self.results = results

    def arrange_plot(self, *plots, shape=None):
        """
        Arrange all specified plots on the same figure
        """
        n = len(plots)
        r = math.sqrt(n)
        if n == 2:
            shape = (1, 2)
        if n == 3:
            shape = (1, 3)
        if shape is None:
            s = int(math.sqrt(n)) + 1            
            if r - int(r) > 0.5:
                shape = (s, s)
            else:
                shape = (s - 1, s)        
            
        fig, axes = plt.subplots(shape[0], shape[1], figsize=(19.2, 10.8))
        axes = axes.flatten()
        res = {}
        for i, plot in enumerate(plots):
            # Separate kwargs if specified
            try:
                plot, kwargs = plot
            except:
                kwargs = {}

            # Create a temp dict containing output kwargs and input kwargs
            temp = copy.deepcopy(res)
            temp.update(kwargs)

            plot_func = getattr(self, plot)            
            res_ = plot_func(axes[i], **temp)
            res.update(res_)

        plt.show()

    def draw_solution(self, ax, minv, minp, xpad, ypad, label_fontsize):
        """
        Adds the solution (vstar and pstar) to an order book plot if they where 
        specified
        """
        vstar, pstar = self.results
        if vstar is not None and pstar is not None:
            ax.scatter(vstar, pstar, marker="X", s=12)

            ax.plot([minv - xpad, vstar], [pstar, pstar], linestyle="--", c='0.8')
            ax.plot([vstar, vstar], [minp - ypad, pstar], linestyle="--", c='0.8')

            vs = "$V^*"
            ps = "$P^*"        
            
            vs += f" = {round(vstar, ndigits=2)}MWh"
            ps += f" = {round(pstar, ndigits=2)}Eur/MWh"
            vs += "$"
            ps += "$"        
            ax.text(vstar, minp + ypad, vs, fontsize=label_fontsize)        
            ax.text(minv + xpad, pstar, ps, fontsize=label_fontsize)        

    def dual_function(self, ax_=None, **kwargs):
        if "pmin" not in kwargs:
            pmin = self.order_book.pmin
        else:
            pmin = kwargs["pmin"]
        if "pmax" not in kwargs:
            pmax = self.order_book.pmax
        else:
            pmax = kwargs["pmax"]
            
        lrange = np.arange(pmin, pmax, (pmax - pmin) / 1000)
        return self.plot_dual_(lrange, ax_=ax_)

    def dual_derivative(self, ax_=None, method="piecewise", **kwargs):
        if "pmin" not in kwargs:
            pmin = self.order_book.pmin
        else:
            pmin = kwargs["pmin"]
        if "pmax" not in kwargs:
            pmax = self.order_book.pmax
        else:
            pmax = kwargs["pmax"]
            
        lrange = np.arange(pmin, pmax, (pmax - pmin) / 1000)        
        return self.plot_dual_der_(lrange, ax_=ax_, method=method, **kwargs)

    def plot_dual_(self, lrange, ax_=None, **kwargs):
        """
        Plot the dual function of the problem. This should be the same for several
        problems.
        """
        if ax_ is None:
            fig, ax = plt.subplots(1)
        else:
            ax = ax_

        duals = [self.solver.dual_function(l) for l in lrange]        

        indmin = np.argmin(duals)
        mind = duals[indmin]
        lambda_opt = lrange[indmin]
        
        if ax is None:
            fig, ax = plt.subplots(1)
            
        ax.plot(lrange, duals)
        ax.set_title("Dual function")
        ax.set_xlabel("$\lambda$")
        ax.set_ylabel("$D(\lambda)$")
        
        maxd = np.max(duals)
        ax.plot([min(lrange), lambda_opt], [mind, mind], linestyle="--", c='0.8')
        ax.plot([lambda_opt, lambda_opt], [mind - 0.1 * (maxd-mind), mind],
                linestyle="--", c='0.8')
        ax.scatter([lambda_opt], [mind], marker="X", s=15)

        lstring = str(round(lambda_opt, ndigits=2)) + "EUR/MWh"
        ax.text(lambda_opt, mind - 0.1 * (maxd-mind), "$\lambda^*$ = " + lstring)
        ax.text(lrange[0], mind, "$D(\lambda^*)$")         
        
        ax.set_xlim([lrange[0], lrange[-1]])
        ax.set_ylim([mind - 0.1 * (maxd-mind), maxd])
        ax.grid("on")
        if ax_ is None:
            plt.show()
        else:
            return {"lambda_dual" : lambda_opt}

    def plot_dual_der_(self, lrange, ax_=None, method="piecewise", **kwargs):
        """
        Plot the derivative of the dual function of the problem. 
        This should be the same for several problems.
        """
        if ax_ is None:
            fig, ax = plt.subplots(1)
        else:
            ax = ax_
            
        duals = np.array([self.solver.dual_derivative_lambda(l, method=method)
                          for l in lrange])

        # Get the closest value to 0
        indmin = np.argmin(np.abs(duals))
        zerod = duals[indmin]
        lambda_opt = lrange[indmin]
            
        if ax is None:
            fig, ax = plt.subplots(1)
            
        ax.plot(lrange, duals)
        ax.set_title(f"Dual derivative using {method}")
        ax.set_xlabel("$\lambda$")
        ax.set_ylabel("$D(\lambda)$")
        
        maxd = np.max(duals)
        mind = np.min(duals)

        ax.plot([lambda_opt, lambda_opt], [mind - 0.1 * (maxd-mind), zerod],
                linestyle="--", c='0.8')
        ax.scatter([lambda_opt], [zerod], marker="X", s=15)        
        ax.plot([min(lrange), lambda_opt],[zerod, zerod],linestyle="--",c='0.8')
        
        lstring = str(round(lambda_opt, ndigits=2)) + "EUR/MWh"        
        ax.text(lambda_opt, mind - 0.1 * (maxd-mind), "$\lambda^*$" +lstring)
        ax.text(lrange[0], mind, "$D'(\lambda^*) = 0$")         
        
        ax.set_xlim([lrange[0], lrange[-1]])
        ax.set_ylim([mind - 0.1 * (maxd-mind), maxd + 0.1 * (maxd-mind)])
        ax.grid("on")
        if ax_ is None:
            plt.show()
        else:
            return {f"lambda_{method}" : lambda_opt}

    def compare_solutions(self, ax_=None, **kwargs):
        if ax_ is None:
            fig, ax = plt.subplots(1)
        else:
            ax = ax_

        methods = []
        results = []
        print(kwargs)
        for method in kwargs.keys():
            if "lambda" in method:
                methods += [method.split("lambda")[1][1:-1]]
                results += [kwargs[method]]

        print(range(len(methods)))
        print(results)
        ax.bar(range(len(methods)), results)
        ax.set_xticks(range(len(methods)), methods, rotation=45)
        ax.set_ylabel("\lambda^* EUR/MWh")

        rmin = min(results)
        rmax = max(results)
        rstep = 0.1 * (rmax - rmin) + 0.01
        ax.set_ylim([rmin - rstep, rmax + rstep])
        ax.grid("on")
        return {}          

        
class BinaryPloter(Ploter):
    def __init__(self, order_book, solver, results):
        Ploter.__init__(self,order_book, solver, results)
        
    def display(self, ax_=None, schema=False,
                linewidth=2, label_fontsize=20, fontsize=30, **kwargs):
        """
        Display the order book and the solution if specified. This function is 
        the same for all Binary problems.
        """
        if ax_ is None:
            fig, ax = plt.subplots(1)
        else:
            ax = ax_

        # Compute plot boundaries
        od = self.order_book.od
        os = self.order_book.os        
        minv = min([od.v0, os.v0])
        maxv = max([od.v0 + od.V, os.v0 + os.V])
        minp = min([od.p0 + od.P, os.p0])
        maxp = max([od.p0, os.p0 + os.P])

        xpad = 10 * (maxv - minv) / 100
        ypad = 10 * (maxp - minp) / 100

        miny = minp + ypad
        minx = minv + xpad 

        ax.set_xlim(minv - xpad, maxv + xpad)
        ax.set_ylim(minp - ypad, maxp + ypad)
        
        # Plot orders
        ax.plot([os.v0, os.v0 + os.V], [os.p0, os.p0 + os.P], marker=".",
                c="b", label="Supply", markersize=12, linewidth=linewidth)
        ax.plot([od.v0, od.v0 + od.V],
                [od.p0, od.p0 + od.P],
                marker=".", c="r", label="Demand",
                markersize=12, linewidth=linewidth)

        # Plot dotted lines
        ax.plot([os.v0, os.v0], [minp - ypad, os.p0],
                linestyle="--", c='0.8')
        ax.plot([os.v0 + os.V, os.v0 + os.V],
                [minp - ypad, os.p0 + os.P], linestyle="--", c='0.8')
        ax.plot([od.v0, od.v0], [minp - ypad, od.p0],
                linestyle="--", c='0.8')
        ax.plot([od.v0 + od.V, od.v0 + od.V],
                [minp - ypad, od.p0 + od.P],linestyle="--", c='0.8')
        
        ax.plot([minv - xpad, os.v0], [os.p0, os.p0],
                linestyle="--", c='0.8')
        ax.plot([minv - xpad, os.v0 + os.V],
                [os.p0 + os.P, os.p0 + os.P],
                linestyle="--", c='0.8')
        ax.plot([minv - xpad, od.v0], [od.p0, od.p0],
                linestyle="--", c='0.8')
        ax.plot([minv - xpad, od.v0 + od.V],
                [od.p0 + od.P, od.p0 + od.P],
                linestyle="--", c='0.8')        
        
        # Place text       
        ax.text(od.v0, minp - ypad, "$V_{D1}$", fontsize=label_fontsize)
        ax.text(od.v0 + od.V, minp - ypad, "$V_{D2}$",
                fontsize=label_fontsize)
        ax.text(os.v0, minp - ypad, "$V_{S1}$", fontsize=label_fontsize) 
        ax.text(os.v0 + os.V, minp - ypad, "$V_{S2}$",
                fontsize=label_fontsize)
        
        ax.text(minv - xpad, od.p0, "$P_{D1}$", fontsize=label_fontsize)
        ax.text(minv - xpad, od.p0 + od.P, "$P_{D2}$",
                fontsize=label_fontsize)
        ax.text(minv - xpad, os.p0, "$P_{S1}$", fontsize=label_fontsize)
        ax.text(minv - xpad, os.p0 + os.P, "$P_{S2}$",
                fontsize=label_fontsize)
        
        ax.text(maxv + xpad, minp - ypad, "$V$ (MWh)", fontsize=label_fontsize)
        ax.text(minv - xpad, maxp + ypad,"$P$ (\euro/MWh)", fontsize=label_fontsize)
        ax.set_title("Order Book")
        ax.legend()    
        
        # Plot arrangement
        if schema:        
            ax.set_xticks([])
            ax.set_yticks([])

            ax.spines["left"].set_position(("data", minv - 1))
            ax.spines["bottom"].set_position(("data", minp - 1))
            ax.spines[["top", "right"]].set_visible(False)
            ax.plot(1, minp - 1, ">k",transform=ax.get_yaxis_transform(),
                    clip_on=False)
            ax.plot(minv - 1, 1, "^k",transform=ax.get_xaxis_transform(),
                    clip_on=False)

        #self.draw_solution(ax, minv, minp, xpad, ypad, label_fontsize)
            
        if ax_ is None:
            plt.show()
        else:
            return {}

            
class BinaryGraphicalPloter(BinaryPloter):
    def __init__(self, order_book, solver, results):
        BinaryPloter.__init__(self, order_book, solver, results)

        
class BinaryAnalyticPloter(BinaryPloter):
    def __init__(self, order_book, solver, results):
        BinaryPloter.__init__(self, order_book, solver, results)

    def social_welfare(self, ax_=None, indices=None, **kwargs):
        if ax_ is None:
            fig, ax = plt.subplots(1)
        else:
            ax = ax_

        acceptance = np.arange(0, 1+0.01, 0.01)            
        social_welfare = np.zeros((len(acceptance), len(acceptance)))
        for i, a_s in enumerate(acceptance):        
            for j, a_d in enumerate(acceptance):
                social_welfare[i, j] = self.solver.social_welfare(a_s, a_d)
        
        im = ax.imshow(social_welfare.transpose(),
                       cmap=plt.get_cmap("Reds"),origin="lower")
        ax.set_title("Social Welfare")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("a_s (%)")
        ax.set_ylabel("a_d (%)")        

        if indices is not None:
            xs, ys = indices
            ax.scatter(xs, ys, marker="X", c="k", s=12)
            
        if ax_ is None:
            plt.show()
        else:
            return {}

    def energy_balance(self, ax_=None, **kwargs):
        if ax_ is None:
            fig, ax = plt.subplots(1)
        else:
            ax = ax_

        acceptance = np.arange(0, 1+0.01, 0.01)            
        energy_balance = np.zeros((len(acceptance), len(acceptance)))
        for i, a_s in enumerate(acceptance):        
            for j, a_d in enumerate(acceptance):
                energy_balance[i, j] = self.solver.energy_balance(a_s, a_d)
        
        vabs = np.max(np.abs(energy_balance))
        xs, ys = np.where(energy_balance==0)
        
        im = ax.imshow(energy_balance.transpose(),cmap=plt.get_cmap("seismic"),
                            origin="lower", vmin=-vabs, vmax=vabs)
        ax.set_title("Energy Balance")        
        plt.colorbar(im, ax=ax)
        
        ax.set_xlabel("a_s (%)")
        ax.set_ylabel("a_d (%)")
        ax.scatter(xs, ys, marker="X", c="k", s=12)

        if ax_ is None:
            plt.show()
        else:
            return {"indices" : (xs, ys)}

    def dual_function(self, ax_=None, **kwargs):
        pmin = self.order_book.pmin
        pmax = self.order_book.pmax
        lrange = np.arange(pmin, pmax, (pmax - pmin) / 1000)        
        return self.plot_dual_(lrange, ax_=ax_)


class SimplePloter(Ploter):
    def __init__(self, order_book, solver, results):
        Ploter.__init__(self, order_book, solver, results)
    
    def display(self, ax_=None, schema=False, colors=None, labels=None,
                linewidth=2, label_fontsize=20, fontsize=30, alpha=0.5, step=0.01,
                fit_to_data=True, **kwargs):
        if ax_ is None:
            fig, ax = plt.subplots(1)
        else:
            ax = ax_

        pmin = self.order_book.pmin
        pmax = self.order_book.pmax

        minv = 0
        maxv = self.order_book.vsum
        xpad = 10 * (maxv - minv) / 100

        supply, demand = self.order_book.curves(
            pmin, pmax, step, fit_to_data=fit_to_data)        
        if not fit_to_data:
            prange = self.order_book.price_range(pmin, pmax, step)
            prange_supply = prange
            prange_demand = prange
        else:
            prange_supply = supply[1]
            prange_demand = demand[1]

            supply = supply[0]
            demand = demand[0] 
            
        delta = 15 * (pmax - pmin) / 100
        pmin -= delta
        pmax += delta
        if colors is None:
            cs = "b"
            cd = "r"
        else:
            cs = colors
            cd = colors            
            
        if labels is None:
            ls = "Supply"
            ld = "Demand"
        else:
            ls = labels
            ld = None
            
        ax.plot(supply, prange_supply, c=cs, label=ls, linewidth=linewidth)
        ax.plot(demand, prange_demand, c=cd, label=ld, linewidth=linewidth)
            
        if labels is None:
            ax.legend(fontsize=fontsize)
            ax.set_ylim([pmin, pmax])
            ax.set_xlim([minv - xpad, maxv + xpad])            
            
        ax.set_xlabel("Cumulated volume (MWh)", fontsize=fontsize)
        ax.set_ylabel("Price (EUR/MWh)", fontsize=fontsize)        
        ax.set_title("Aggregated Curves", fontsize=fontsize)    
        ax.grid("on")        

        if ax_ is None:
            plt.show()
        else:
            return {}
        
    def plot_derivative_order(self, what, variable, order,
                                   both=False, ax_=None, **kwargs):
        """
        Plots the derivative of 'what' function with respect one 'variable', for 
        the specified 'order'.
        """
        if ax_ is None:
            fig, ax = plt.subplots(1)
        else:
            ax = ax_

        i = order
        func_name = f"{what}_derivative_{variable}"
        func = getattr(self.solver, func_name)
            
        if i > len(self.order_book.orders) - 1:
            raise Exception("Please specify an order in the range!")

        step = 0.01
        pmin = self.order_book.pmin
        pmax = self.order_book.pmax        
        prange = self.order_book.price_range(pmin, pmax, step)
        der = [func(i, p) for p in prange]        
        ax.plot(prange, der, label=variable)

        o = self.order_book.orders[i]
        ax.set_title(f"{str(o)}")
        ax.set_ylabel(f"d {what}/d {variable}")
        ax.set_xlabel("lambda")
        ax.grid("on")
        
        if ax_ is None:
            plt.show()
        else:
            return {}

    def plot_derivative(self, what, variable, both=False, ax_=None, **kwargs): 
        n = len(self.order_book.orders)
        sq = int(math.sqrt(n)) + 1
        shape = (sq, sq)
        if n == 2:
            shape = (1, 2)
        if n == 3:
            shape = (1, 3)        
            
        if ax_ is None:
            fig, ax = plt.subplots(shape[0], shape[1], sharex=True, sharey=True)
            ax = np.array(ax).flatten()
        else:
            ax = ax_

        for i in range(n):
            self.plot_derivative_order(what, variable, i, both=both, ax_=ax[i])

        if ax_ is None:
            fig.suptitle(f"Derivative of {what} with respect to {variable}")
            plt.show()
        else:
            ax[-1].legend()
            return {}

    def lambda_star(self, ax_=None, label_fontsize=10, **kwargs):
        if ax_ is None:
            fig, ax = plt.subplots(1)
        else:
            ax = ax_
           
        step = 0.01
        pmin = self.order_book.pmin
        pmax = self.order_book.pmax        
        prange = self.order_book.price_range(pmin, pmax, step)
        der = [self.solver.lambda_star(p) for p in prange]        
        ax.plot(prange, der)
        ax.plot(prange, prange)

        ind = np.where(np.abs(prange-der) < 0.01)[0][0]
        pstar = prange[ind]
        lstar = der[ind]
        lmin = min(der)
        lmax = max(der)
        xpad = 0
        ypad = 0
        ax.set_ylim([lmin - ypad, lmax + ypad])
        ax.set_xlim([pmin - xpad, pmax + xpad])

        ax.scatter(pstar, lstar, marker="X", s=20, c="k")
        ax.plot([pstar, pstar], [lmin - ypad, lstar], linestyle="--", c='0.8')
        ax.plot([pmin - xpad, pstar], [lstar, lstar], linestyle="--", c='0.8')

        ps = str(round(pstar, ndigits=2)) + "EUR/MWh"
        vs = str(round(lstar, ndigits=2)) + "EUR/MWh"
        ax.text(pstar, lmin + ypad, ps, fontsize=label_fontsize)        
        ax.text(pmin + xpad, lstar, vs, fontsize=label_fontsize)
        
        ax.set_title("Lambda*(l)")
        ax.set_ylabel("lambda*")
        ax.set_xlabel("lambda")
        ax.grid("on")
        
        if ax_ is None:
            plt.show()
        else:
            return {}        

        
class MultiplePlotter(object):
    def __init__(self, spliter, X,n_epochs,save_to_disk="",batch_size=30, OBs=None):
        Xt, Xv = spliter(X)
        self.nt = Xt.shape[0]
        self.nv = Xv.shape[0]        
        
        self.save_to_disk = save_to_disk
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        self.OBs = OBs

    def n(self, dataset):
        if dataset == "train":
            return self.nt
        if dataset == "validation":
            return self.nv
        return -1

    def get_path(self, e, b, dataset):
        return os.path.join(
            self.save_to_disk, "epoch_" + str(e), f"{dataset}_batch_" + str(b))

    def get_OBhat(self, e, d, h, dataset):
        if e != -1:
            batch = d // self.batch_size
            d_ = d % self.batch_size
            
            if batch != 0:
                raise Exception("Only the order book of the first batch is saved")
            
            path = os.path.join(self.get_path(e, batch, dataset), "OBhat.npy")
            data = np.load(path)[24 * d_ + h]
        else:
            d_ = d-self.n(dataset)
            path = os.path.join(self.save_to_disk, "OBvhat.npy")
            data = np.load(path)[d_, h]
        OB = TorchOrderBook(data)            
        return OB

    def get_yhat(self, dataset,  e, d, h):
        if dataset == "validation":
            raise Exception(f"We don't have access to the {dataset} forecasts!")
        if e != -1:
            batch = d // self.batch_size
            d_ = d % self.batch_size

            if batch != 0:
                raise Exception("Only the order book of the first batch is saved")
            
            path = os.path.join(self.get_path(e, batch, dataset), "yhat.npy")
            yhat = np.load(path)[d_, h]
        else:
            d_ = d - self.n(dataset)
            path = os.path.join(self.save_to_disk, "yvhat.npy")
            yhat = np.load(path)[d_, h]
        return yhat

    def get_all_yhat(self, dataset, e):
        n_batches = (self.n(dataset)// self.batch_size) + 1            
        values = np.zeros((self.n(dataset), 24))
        current = 0
        for b in range(n_batches):
            path = self.get_path(e, b, dataset)
            path_ = os.path.join(path, "yhat.npy")
            batch = np.load(path_)
            
            batch_reshaped = batch.reshape(-1, 24)
            bs = batch_reshaped.shape[0]
            values[current:current+bs] = batch_reshaped
            current += bs
        return values

    def get_variable(self, variable, e, dataset):
        if e != -1:
            n_batches = (self.n(dataset)// self.batch_size) + 1            
            values = np.zeros((self.n(dataset), 24, self.OBs))
            current = 0
            for b in range(n_batches):
                path = self.get_path(e, b, dataset)
                path_ = os.path.join(path, variable + ".npy")
                batch = np.load(path_)
                
                batch_reshaped = batch.reshape(-1, 24, self.OBs)
                bs = batch_reshaped.shape[0]
                values[current:current+bs] = batch_reshaped
                current += bs
        else:
            path = os.path.join(self.save_to_disk, "OBvhat.npy")
            if "V" in variable:
                ind = 0
            elif "Po" in variable:
                ind = 1
            else:
                ind = 2
            values = np.load(path)[:, ind]
                
        return values
           
    def display(self, d, h, dataset, ax_=None, linewidth=2, fontsize=30,
                colormap='copper_r', epochs=None, **kwargs):
        if ax_ is None:
            fig, ax = plt.subplots(1, figsize=(19.2, 10.8))
        else:
            ax = ax_

        cmap = plt.get_cmap(colormap)
        if epochs is None:
            epochs = range(self.n_epochs)
            
        for i, e in enumerate(epochs):
            OB = self.get_OBhat(e, d, h, dataset)            
            yhat = self.get_yhat(e, d, h, dataset)
            
            ploter = get_ploter(OB)
            if e != -1:
                color_index = (i + 1) / (len(epochs) + 1)
                color = cmap(color_index)
            else:
                color = "r"

            v = OB.accepted_volume(yhat)
            ax.scatter(v, yhat, marker="X", s=200, color=color)
            ploter.display(ax_=ax, fit_to_data=False, linewidth=linewidth,alpha=0.5,
                           colors=color, labels=f"Epoch {e}", fontsize=fontsize)
        ax.legend(fontsize=fontsize)
        
        if ax_ is None:
            plt.show()
        else:
            return {}
        
    def price_forecasts_distributions(self, dataset, e, ax_=None,
                                      linewidth=2, label_fontsize=20, 
                                      fontsize=30, colormap='hsv', **kwargs):
        if ax_ is None:
            fig, ax = plt.subplots(1)
        else:
            ax = ax_

        yhat = self.get_all_yhat(dataset, e).reshape(-1)
            
        color = "b"
        ax.hist(yhat.reshape(-1), histtype='step', bins=100, color=color,
                linewidth=linewidth)        
        ax.grid("on")
        
        if ax_ is None:
            plt.show()
        else:
            return {}
        
        
    def distribution(self, dataset, ax_=None, steps=-1, epochs=None, variables=-1,
                     linewidth=2, label_fontsize=20, 
                     fontsize=30, colormap='hsv', **kwargs):
        cmap = plt.get_cmap(colormap)

        ylabels = ["Out of base NN", "After Scaling", "After signs"]
        base_variables = np.array(["V", "Po", "PoP", "P"])
        if epochs is None:
            epochs = range(self.n_epochs)
            
        if steps==-1:
            steps = [1, 2, 3]
        elif type(steps).__name__ != 'list':
            steps = [steps]
            
        if variables!=-1:
            base_variables = base_variables[variables]

        if ax_ is None:
            fig, axes = plt.subplots(len(steps), len(base_variables),
                                     sharey="row", sharex="col",
                                     gridspec_kw={"hspace" : 0.0, "wspace" : 0.0})
            axes = axes.reshape(len(steps), len(base_variables))
        else:
            axes = ax_

        for x, step in enumerate(steps):
            axes[x, 0].set_ylabel(ylabels[step - 1])
            variables = [f"{v}{step}" for v in base_variables]
            for y, variable in enumerate(variables):
                for i, e in enumerate(epochs):
                    values = self.get_variable(variable, e, dataset)
                    color_index = (i + 1) / (len(epochs) + 1)
                    axes[x, y].hist(
                        values.reshape(-1), bins=500, histtype="step",
                        color=cmap(color_index), label=f"Epoch {e}",
                        linewidth=linewidth)
                    axes[x, y].tick_params(labelsize=fontsize)
                
                axes[x, y].set_title(variable, y=0.85)
                axes[x, y].grid("on")

        axes[x, y].legend()
        plt.suptitle("Variable distribution accross epochs")
        
        if ax_ is None:
            plt.show(block=False)
        else:
            return {}

    def get_extreme_orders(self, dataset, steps=-1, epochs=None, variables=-1):
        ylabels = ["Out of base NN", "After Scaling", "After signs"]
        base_variables = np.array(["V", "Po", "PoP", "P"])
        if epochs is None:
            epochs = range(self.n_epochs)
            
        if steps==-1:
            steps = [1, 2, 3]
        elif type(steps).__name__ != 'list':
            steps = [steps]
            
        if variables!=-1:
            base_variables = base_variables[variables]

        res = np.zeros((len(epochs), len(base_variables), len(steps),
                        self.n(dataset), 24, 2))
        for x, step in enumerate(steps):
            variables = [f"{v}{step}" for v in base_variables]
            for y, variable in enumerate(variables):
                for i, e in enumerate(epochs):
                    values = self.get_variable(variable, e, dataset)
                    res[i, y, x, :, :, 0] = values[:, :, 0]
                    res[i, y, x, :, :, 1] = values[:, :, -1]
                    
        return res

    def plot_extreme_orders(self, dataset, steps=-1, epochs=0, variables=1, d=None,
                            h=None, ax_=None, linewidth=2, label_fontsize=20, 
                            fontsize=30):
        res = self.get_extreme_orders(dataset, steps=-1, epochs=None,
                                      variables=-1)
        if d is None:
            d = range(self.n(dataset))
        if h is None:
            h = range(23)        
        
        ylabels = ["Out of base NN", "After Scaling", "After signs"]
        base_variables = np.array(["V", "Po", "PoP", "P"])

        if variables!=-1:
            base_variables = base_variables[variables]        

        if steps==-1:
            steps = [1, 2, 3]
        elif type(steps).__name__ != 'list':
            steps = [steps]
        
        cmap_min = plt.get_cmap("hsv")
        cmap_max = plt.get_cmap("hsv")
        if ((type(d).__name__ != 'int') and (len(base_variables) == 1)):
            variable = base_variables[0]
            if ax_ is None:
                fig, (ax, axmax) = plt.subplots(2, 1, gridspec_kw={"hspace" : 0.0})
            else:
                ax, axmax = ax_

            markers = ["o", "v", "^"]
            for s in steps:
                data_min = res[epochs, variable, s-1, :, :, 0].reshape(-1)
                color_index = (s-1 + 1) / (len(steps) + 1)
                color = cmap_min(color_index)
                
                xindices = range(len(data_min))
                ax.scatter(xindices, data_min, c=color, label=ylabels[s-1],
                           marker=markers[s-1], alpha=0.6)

                data_max = res[epochs, variable, s-1, :, :, 1].reshape(-1)
                color_index = (s-1 + 1) / (len(steps) + 1)
                color = cmap_max(color_index)
                axmax.scatter(xindices, data_max, c=color, label=ylabels[s-1],
                              marker=markers[s-1], alpha=0.6)

            ax.set_ylabel("Po min")
            axmax.set_ylabel("Po max")
            ax.legend()
            axmax.legend()            
        else:
            if ax_ is None:
                fig, axes = plt.subplots(2, 
                    len(base_variables),
                    gridspec_kw={"hspace" : 0.0})
                for i, variable in enumerate(base_variables):
                    for s in steps:
                        ax = axes[0, i]
                        data_min = res[epochs, i, s-1, d, h, 0]
                        color_index = (s-1 + 1) / (len(steps) + 1)
                        color = cmap_min(color_index)
                    
                        xindices = s
                        ax.bar(xindices,data_min,width=1,color=color,
                               label=ylabels[s-1], edgecolor="k", alpha=0.6)
                        ax.grid("on", axis="y")
                        ax.set_title(variable, y=0.85)
                        ax.set_xticks([])

                        axmax = axes[1, i]                        
                        data_max = res[epochs, i, s-1, d, h, 1].reshape(-1)
                        color_index = (s-1 + 1) / (len(steps) + 1)
                        color = cmap_max(color_index)
                        axmax.bar(xindices,data_max,width=1,color=color,
                                  label=ylabels[s-1], edgecolor="k", alpha=0.6)
                        axmax.grid("on", axis="y")
                        axmax.set_xticks([])                        
                    
                    for ax in axes[0, [1, 2]].flatten():
                        ax.set_ylim([-490, -510])
                        
                    for ax in axes[1, [1, 2]].flatten():                    
                        ax.set_ylim([2950, 3050])
                        
                    axes[0, 0].set_ylabel("PMIN")                        
                    axes[1, 0].set_ylabel("PMAX")

                    ax.legend()

                plt.suptitle(
                    f"Limit orders at epoch {epochs} for day {d}, hour {h}")
            else:
                ax, axmax = ax_            
        
        if ax_ is None:
            plt.show()
        else:
            return {}

    def scaling_summary(self, regr, Y, linewidth=2, fontsize=30):
        dataset = "train"
        e = 0
        fig, axes = plt.subplots(3, 2, figsize=(19.2, 10.8),
                                  gridspec_kw={"wspace" : 0.0})
        
        # Plot forecasted OB distirbution
        variables = ["Po", "PoP", "P"]
        for i in range(3):
            variable = f"{variables[i]}3"
            ax = axes[i, 0]
            values = self.get_variable(variable, e, dataset)
            ax.hist(values.reshape(-1), bins=500, histtype="step",
                    color="b", label=f"Epoch {e}",
                    linewidth=linewidth)
            ax.set_title(variable, y=0.85)
            ax.grid("on")

        # Plot forecasted prices distirbution
        ax = axes[1, 1]
        self.price_forecasts_distributions(dataset, e, ax_=ax)
        ax.set_title("Yhat", y=0.85)
        
        # Plot transformed prices distirbution
        ax = axes[2, 1]
        ax.set_title("Ytransformed", y=0.85)
        Yt = regr.steps[1][1].transformer.transform(Y).reshape(-1) 
        ax.hist(Yt.reshape(-1), histtype='step', bins=100, color="b",
                linewidth=linewidth)
        ax.grid("on")

        # Signs analysis
        ax = axes[0, 1]
        self.limit_orders(e, dataset, ax_=ax, fontsize=fontsize)
        
        # Format title
        transformer = regr.steps[1][1].transformer.scaling
        po_cliper = str(regr.steps[1][1].model.Po_scaler)
        pop_cliper = str(regr.steps[1][1].model.PoP_scaler)
        signs = str(regr.steps[1][1].model.sign_layer)
        wis = [wi._str(round_=True)
               for wi in regr.steps[1][1].model.weight_initializers]
        
        axes[0, 0].text(0.6, 0.5, po_cliper, transform=axes[0, 0].transAxes)
        axes[1, 0].text(0.6, 0.5, pop_cliper, transform=axes[1, 0].transAxes)
        axes[2, 0].text(0.6, 0.5, signs, transform=axes[2, 0].transAxes)
        axes[2, 1].text(0.75, 0.5, transformer, transform=axes[2, 1].transAxes)
        for i, wi in enumerate(wis):
            axes[0, 0].text(0.6, 0.5 - 0.1 * (i + 1), wi,
                            transform=axes[0, 0].transAxes)
            axes[1, 0].text(0.6, 0.5 - 0.1 * (i + 1), wi,
                            transform=axes[1, 0].transAxes)

        
        title = "Scaling summary "        
        plt.suptitle("Scaling summary")
        plt.show()
        

    def limit_orders(self, e, dataset, ax_=None, fontsize=20):
        if ax_ is None:
            fig, ax = plt.subplots(1)
        else:
            ax = ax_

        V = self.get_variable("V3", e, dataset)
        P = self.get_variable("P3", e, dataset)

        p_diff = 100 * np.mean(
            np.logical_and(np.sign(P) != np.sign(V),
                           (np.sign(P) != 0)))

        lps = P[:, :, 0].reshape(-1)
        lpd = P[:, :, -1].reshape(-1)
       
        lvs = V[:, :, 0].reshape(-1)
        lvd = V[:, :, -1].reshape(-1)
        
        frac_s = 100 * np.mean(np.logical_and((lps >= 0), (lvs > 0)))
        frac_d = 100 * np.mean(np.logical_and((lpd <= 0), (lvd < 0)))

        b1 = ax.bar([1], [p_diff],color="r", edgecolor="k",
                    label="% of sign difference", width = 0.75)
        b2 = ax.bar([2], [frac_s],color="b",edgecolor="k", width = 0.75, 
                    label="% of correct supply limit orders")
        b3 = ax.bar([3], [frac_d], color="g", edgecolor="k", width = 0.75, 
                    label="% of correct demand limit orders")
        ax.legend()

        for (b, t) in zip([b1, b2, b3], [p_diff, frac_s, frac_d]):
            y = b.patches[0].get_y() + b.patches[0].get_height()
            x = b.patches[0].get_x()
            txt = str(round(t, ndigits=2)) + "%"
            ax.text(x, y, txt, fontsize=fontsize)            

        ax.grid("on", axis="y")
            
        if ax_ is None:
            plt.show()
        else:
            return {}            
 
        
class ExpPloter(object):
    def __init__(self, ploters, save_path):
        # A dict of ploters
        self.ploters = ploters
        self.save_path = save_path

    def price_forecasts(self, Yv, fontsize=20):
        fig, axes = plt.subplots(
            4, 2, figsize=(19.2, 10.8), sharex=True, sharey=True,
            gridspec_kw={"wspace" : 0.0, "hspace" : 0.0})
        axes = axes.flatten()

        for name, ax in zip(self.ploters.keys(), axes):
            yvhat = np.load(os.path.join(self.save_path, name, "yvhat.npy"))
            ax.plot(yvhat.reshape(-1), c="r")
            ax.plot(Yv.reshape(-1), c="b")
            ax.grid("on")
            ax.set_xticklabels([])

        # Add a legend
        blue, = ax.plot([1, 1], c="b")
        red, = ax.plot([1, 1], c="r")
        fig.legend([blue, red], ['$Y$', '$\widehat{Y}$'], fontsize=fontsize,
                   ncols=2, loc=8)

        # Set lines labels
        axes[0].set_ylabel("No scaling", fontsize=fontsize)
        axes[2].set_ylabel("Scaling $\widehat{OB}$", fontsize=fontsize)
        axes[4].set_ylabel("Weight Initalization", fontsize=fontsize)

        # Set columns labels
        axes[0].set_title("$Y$", fontsize=fontsize)
        axes[1].set_title("$Y_{transformed}$", fontsize=fontsize)

    def price_forecasts_distributions(self, Yv, fontsize=20, linewidth=4):
        fig, axes = plt.subplots(
            1, 4, figsize=(19.2, 10.8))
        axes = axes.flatten()                
        for name in self.ploters.keys():
            ax = axes[int(name[1]) - 1]
            
            yvhat = np.load(os.path.join(self.save_path, name, "yvhat.npy"))
            color = ["b", "r"][int(name[-1]) - 1]
            ax.hist(yvhat.reshape(-1), histtype='step', bins=100, color=color,
                    linewidth=linewidth)
            ax.hist(Yv.reshape(-1), histtype='step', bins=100, color="g",
                    linewidth=linewidth)            
            ax.grid("on")
            ax.set_yticklabels([])
            
        axes[0].set_title("No scaling", fontsize=fontsize)
        axes[1].set_title("Scaling $\widehat{OB}$", fontsize=fontsize)
        axes[2].set_title("Weight Initalization", fontsize=fontsize)            

        # Add a legend
        blue, = ax.plot([1, 1], c="b", linewidth=linewidth)
        red, = ax.plot([1, 1], c="r", linewidth=linewidth)
        green, = ax.plot([1, 1], c="g", linewidth=linewidth)
        fig.legend([blue, red, green], ["$Y$", "$Y_{transformed}$", "Real Prices"],
                   fontsize=fontsize, ncols=3, loc=8)

    def distribution(self, distribution, fontsize=20, linewidth=4):
        fig, axes = plt.subplots(4, 2, figsize=(19.2, 10.8),
                                 gridspec_kw={"wspace" : 0.0})

        colormaps = ["Blues", "Reds"]
        for p in self.ploters.keys():
            line = int(p[1])
            color = colormaps[int(p[-1]) - 1]
            axes_line = axes[line-1, :].reshape(1, -1)
            
            ploter = self.ploters[p]
            
            # Plot tensor distirbution across epochs
            ploter.distribution(distribution,
                epochs=[0], steps=3, variables=[1, 3], colormap=color,
                ax_=axes_line, fontsize=fontsize, linewidth=linewidth)

        # Correct things for the plot
        axes = axes.flatten()
        for ax in axes:
            ax.set_yticklabels([])
            ax.set_title("")
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

        # Set lines labels
        axes[0].set_ylabel("No scaling", fontsize=fontsize)
        axes[2].set_ylabel("Scaling $\widehat{OB}$", fontsize=fontsize)
        axes[4].set_ylabel("Weight Initalization", fontsize=fontsize)

        # Set columns labels
        axes[0].set_title("Po", fontsize=fontsize)
        axes[1].set_title("P", fontsize=fontsize)

        # Add a legend
        blue, = ax.plot([1, 1], c=plt.get_cmap(colormaps[0])(0.5))
        red, = ax.plot([1, 1], c=plt.get_cmap(colormaps[1])(0.5))
        ax.legend([blue, red], ['$Y$', '$Y_{transformed}$'], fontsize=fontsize)

        
class ExamplePloter(object):
    def __init__(self, OBs):
        self.OBs = OBs

    def display(self): 
        fig, axes = plt.subplots(1, len(self.OBs), sharey=True,
                                 gridspec_kw={"wspace" : 0.0})
        for OB, ax in zip(self.OBs, axes):
            get_ploter(OB).display(ax_=ax)
                    
        plt.show()
      
        

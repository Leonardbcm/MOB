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
                linewidth=2, label_fontsize=20, fontsize=30,
                fit_to_data=True, **kwargs):
        if ax_ is None:
            fig, ax = plt.subplots(1)
        else:
            ax = ax_

        step = 0.01
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
            
        ax.plot(supply, prange_supply, c=cs, label=ls)
        ax.plot(demand, prange_demand, c=cd, label=ld)
        
        if labels is None:
            ax.legend()
            ax.set_ylim([pmin, pmax])
            ax.set_xlim([minv - xpad, maxv + xpad])            
            
        ax.set_xlabel("Cumulated volume (MWh)")
        ax.set_ylabel("Price (EUR/MWh)")        
        ax.set_title("Aggregated Curves")    
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
    def __init__(self, regr, X):
        self.regr = regr
        Xt, Xv = regr.steps[1][1].spliter(X)
        self.X = Xt
        
        # 0) Get the callback
        self.cb = regr.steps[1][1].callbacks[1]
        self.n_epochs = len(self.cb.OBhats) - 1
        self.n = self.X.shape[0]
        self.OBs = self.cb.OBhats[0][0].shape[1]
        
        # 1) reshape to n_epochs * n_batchs * n_per_batch * OBs * 3
        self.OBhat = np.zeros((self.n_epochs, self.n, 24, self.OBs, 3))
        self.yhat = np.zeros((self.n_epochs, self.n, 24))        
        for i in range(self.n_epochs):
            ep = self.cb.OBhats[i]
            current = 0
            for j, batch in enumerate(ep):
                # Store OB
                batch_reshaped = batch.reshape(-1, 24, self.OBs, 3)
                bs = batch_reshaped.shape[0]
                self.OBhat[i, current:current+bs] = batch_reshaped

                # Store forecast
                yhat = self.cb.yhats[i][j]
                self.yhat[i, current:current+bs] = yhat
                
                current += bs
           
    def display(self, d, h, ax_=None, linewidth=2, label_fontsize=20, fontsize=30,
                colormap='copper_r', epochs=-1, **kwargs):
        if ax_ is None:
            fig, (ax, axp) = plt.subplots(2)
        else:
            ax = ax_

        cmap = plt.get_cmap(colormap)
        if epochs==-1:
            epochs = range(self.n_epochs)
            
        for i, e in enumerate(epochs):
            OB = TorchOrderBook(self.OBhat[e, d, h])
            ploter = get_ploter(OB)

            color_index = (i + 1) / (len(epochs) + 1)
            ploter.display(ax_=ax, fit_to_data=False,
                           colors=cmap(color_index), labels=f"Epoch {e}")
        ax.legend()
        axp.plot(epochs, self.yhat[epochs, d, h], marker=".", markersize=10)
        axp.set_ylabel("Price forecast")
        axp.set_xlabel("Epochs")
        axp.grid("on")
        
        if ax_ is None:
            plt.show()
        else:
            return {}

        
class ExamplePloter(object):
    def __init__(self, OBs):
        self.OBs = OBs

    def display(self): 
        fig, axes = plt.subplots(1, len(self.OBs), sharey=True,
                                 gridspec_kw={"wspace" : 0.0})
        for OB, ax in zip(self.OBs, axes):
            get_ploter(OB).display(ax_=ax)
                    
        plt.show()
      
        

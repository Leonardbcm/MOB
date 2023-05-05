from src.euphemia.order_books import *
from src.analysis.evaluate import ACC, smape
import numpy as np

def compute_losses(model_wrapper, err_func, n=None):
    Yv, Yhatv, Ypo, Yhatpo, Yp, Yhatp, past_prices, real_prices, datetimes = model_wrapper.load_and_reshape()
    if n is None:
        n = Yv.shape[0]

    # Compute duals and losses
    duals = np.zeros(n)
    for idx, dt in enumerate(datetimes[:n]):        
        ref_price = real_prices[idx]
        OB = TorchOrderBook(np.concatenate(
            (Yv[idx].reshape(-1, 1),
             Ypo[idx].reshape(-1, 1),
             Yp[idx].reshape(-1, 1)), axis=1))
        duals[idx] = np.array([o.dual_function(ref_price) for o in OB.orders]).sum()
    
    Vloss = err_func(Yv[1:n], Yhatv[1:n])
    Ploss = err_func(Yp[1:n], Yhatp[1:n])
    Poloss = err_func(Ypo[1:n], Yhatpo[1:n])

    dual_loss = err_func(duals[1:n], duals[0:n-1])
    price_loss = err_func(real_prices[1:n], past_prices[1:n])
    
    return Vloss, Ploss, Poloss, dual_loss, price_loss

def plot_losses(Vloss, Ploss, Poloss, dual_loss, price_loss, OBs):
    fig, ax = plt.subplots(1, figsize=(19.2, 10.8))
    ax.plot(range(n-1), price_loss, label = "Past Price Error", c="k", linewidth=4)
    ax.plot(range(n-1), dual_loss, label = "Dual Difference Error", c="orange",
            linewidth=4)

    ax.plot(range(n-1), Vloss.mean(axis=1), label = "V", c="r", linewidth=4)
    ax.plot(range(n-1), Ploss.mean(axis=1), label = "P", c="b", linewidth=4)
    ax.plot(range(n-1), Poloss.mean(axis=1), label = "Po", c="g", linewidth=4)
    
    for cmap, loss in zip(["Reds", "Blues", "Greens"], [Vloss, Ploss, Poloss]):
        c = plt.get_cmap(cmap)
        for i in range(OBs):
            color_index = (i + 1) / (OBs + 1)
            ax.plot(range(n-1), loss[:, i], c=c(color_index), alpha=0.2)    

    ax.legend()
    plt.show()
    
def plot_corrcoef(coefs, OBs):
    fig, ax = plt.subplots(1, figsize=(19.2, 10.8))
    for i, (cmap, label) in enumerate(zip(["r", "b", "g"], ["V", "Po", "P"])):
        data = coefs[i*(OBs+1):(i+1)*(OBs+1), -1]
        xindices = range(i*(OBs+2), (i+1)*(OBs+1)+i)
        ax.bar(xindices, data, width=0.8, edgecolor="k", color=cmap, label=label)
        
    ax.bar(3 * OBs + 7, coefs[-2, -1], width=0.8, edgecolor="k", color="orange",
           label="Dual Diff")
    ax.legend()
    ax.grid("on")
    ax.set_ylabel("Corr Coeff")
    ax.set_title(f"Error correlation for OBs={OBs}")
    plt.show()
    

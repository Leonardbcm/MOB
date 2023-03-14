import matplotlib.pyplot as plt, numpy as np, os, pandas, matplotlib

def price_distribution(po_bins, po_values, p_bins, p_values, pop_bins, pop_values,
                       fontsize=20, plot_approx=False):
    fig, [ax1, ax3, ax2] = plt.subplots(
        3, 1, figsize=(19.2, 10.8), gridspec_kw={"wspace" : 0.0}, sharex=True)

    ax1.bar(po_bins[1:], po_values, width=po_bins[1] - po_bins[0], color="b",
            label="Po")
    ax1.set_title("Distribution of Po", fontsize=fontsize)
    ax1.grid("on")
    ax1.tick_params(labelsize=fontsize)

    if plot_approx:
        po1 = -65
        po2 = 115
        indices = np.where(np.logical_and(po_bins > po1, po_bins < po2))[0]
        mu = np.average(po_bins[indices], weights=po_values[indices])    
        sigma = np.sqrt(np.cov(po_bins[indices], aweights=po_values[indices]))
        ax1.plot(
            po_bins[indices],
            np.sum(po_values[indices])/(sigma * np.sqrt(2 * np.pi)) *
            np.exp( - (po_bins[indices] - mu)**2 / (2 * sigma**2) ),
            linewidth=2, color='r',
            label=f"N({round(mu, ndigits=2)}, {round(sigma, ndigits=2)})")
        
    ax2.bar(p_bins[1:], p_values, width=p_bins[1] - p_bins[0],  color="g",
            log=True, label="P")
    ax2.grid("on")    
    ax2.set_title("Log distribution of P", fontsize=fontsize)
    ax2.tick_params(labelsize=fontsize)

    if plot_approx:
        pop1 = -65
        pop2 = 115
        indices = np.where(np.logical_and(pop_bins > pop1, pop_bins < pop2))[0]
        mu = np.average(pop_bins[indices], weights=pop_values[indices])    
        sigma = np.sqrt(np.cov(pop_bins[indices], aweights=pop_values[indices]))
        ax3.plot(
            pop_bins[indices],
            np.sum(pop_values[indices])/(sigma * np.sqrt(2 * np.pi)) *
            np.exp( - (pop_bins[indices] - mu)**2 / (2 * sigma**2) ),
            linewidth=2, color='r',
            label=f"N({round(mu, ndigits=2)}, {round(sigma, ndigits=2)})")
        
    ax3.bar(pop_bins[1:], pop_values, width=pop_bins[1] - pop_bins[0], color="c",
            label="Po + P")
    ax3.grid("on")
    if plot_approx:
        ax1.legend(fontsize=fontsize)
        ax2.legend(fontsize=fontsize)                
        ax3.legend(fontsize=fontsize)        
    ax3.set_title("Distribution of Po + P", fontsize=fontsize)
    ax3.tick_params(labelsize=fontsize)    

def number_of_orders(results, fontsize=20):
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(19.2,10.8),
                                   gridspec_kw={"wspace" : 0.0})
    colors = ("r", "b", "g")
    for column, color in zip(results.columns, colors):    
        ax1.hist(results.loc[:, column].values, bins=100, color=color,
                 edgecolor="k",
                 alpha=0.6, label=column) 
        ax2.hist(100 * results.loc[:, column].values / results.values.sum(axis=1),
                 bins=100,color=color,edgecolor="k",alpha=0.6,label=column)

    ax1.hist(results.values.sum(axis=1), bins=100, color="y",
             edgecolor="k", alpha=0.3, label="Total") 
    
    ax1.set_title("Number of orders", fontsize=fontsize)
    ax2.set_title("\% of orders", fontsize=fontsize)
    ax1.grid("on")
    ax2.grid("on")

    ax1.tick_params(axis="both", labelsize=fontsize)
    ax2.tick_params(axis="both", labelsize=fontsize)    
    ax1.legend(fontsize=fontsize)    

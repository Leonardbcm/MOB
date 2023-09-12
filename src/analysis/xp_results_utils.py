import matplotlib.pyplot as plt, torch
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import numpy as np, pandas, copy, datetime
import matplotlib.dates as mdates
from matplotlib.ticker import Formatter

from src.models.spliter import MySpliter
from src.models.torch_wrapper import OBNWrapper
from src.analysis.evaluate import DM, mae
from src.euphemia.ploters import get_ploter
from src.analysis.utils import load_real_prices
from src.models.torch_models.ob_datasets import OrderBookDataset
from src.euphemia.order_books import *
from src.euphemia.solvers import *

def cmap():
    c = 40
    red = np.concatenate([np.linspace(0, 1, c), np.linspace(1, 0.5,100-c)[1:], [0]])
    green = np.concatenate([np.linspace(0.5, 1, c), np.zeros(100-c)])
    blue = np.zeros(100)
    rgb_color_map = np.concatenate([
        red.reshape(-1, 1),
        green.reshape(-1, 1), 
        blue.reshape(-1, 1)], axis=1)
    rgb_color_map = mpl.colors.ListedColormap(rgb_color_map)
    return rgb_color_map

def retrieve_results_OBs(IDs, countries, datasets, OB_sizes, N_VAL, N_SAMPLES,
                         folder, nh = 24, version=None):
    predicted_prices = {}
    real_prices = {}
    predicted_OB = {}
    real_OB = {}
    results = {}
    for OBs in OB_sizes:
        pp, rp, pOB, rOB, r = retrieve_results(
            IDs, countries, datasets, OBs, N_VAL, N_SAMPLES,folder, nh = 24,
            version=version)
        predicted_prices[str(OBs)] = pp
        real_prices[str(OBs)] = rp
        predicted_OB[str(OBs)] = pOB
        real_OB[str(OBs)] = rOB
        results[str(OBs)] = r

    return predicted_prices, real_prices, predicted_OB, real_OB, results
        
def retrieve_results(IDs, countries, datasets, OBs, N_VAL,N_SAMPLES,folder,nh = 24,
                     version=None):
    ####### Results container
    results = pandas.DataFrame(
        columns=[
            "country", "ID",
            "val_price_mae", "val_price_smape", "val_price_ACC", "val_OB_smape",
            "val_OB_ACC"])
    
    nc = len(countries)
    n = len(IDs)    

    real_prices = np.zeros((nc, n, N_VAL, nh))
    real_OB =  np.zeros((nc, n, N_VAL, 72*OBs))

    predicted_prices = np.zeros((nc, n, N_VAL, nh))
    predicted_OB = np.zeros((nc,n, N_VAL, 72*OBs))

    for j, (country, dataset) in enumerate(zip(countries, datasets)):
        for i, ID in enumerate(IDs):

            ###### Create Model wrapper
            spliter = MySpliter(N_VAL, shuffle=False)        
            model_wrapper = OBNWrapper(
                "RESULTS", dataset, spliter=spliter, country=country,
                skip_connection=True, use_order_books=False,
                order_book_size=OBs, IDn=ID, tboard=folder)
            if version is None:
                version_ = model_wrapper.highest_version
            else:
                version_ = f"version_{version}"

            ###### Load DATA        
            X, Y = model_wrapper.load_train_dataset()
            X = X[:N_SAMPLES, :]
            Y = Y[:N_SAMPLES, :]
            (_, _), (Xv, Yv) = model_wrapper.spliter(X, Y)
            
            ###### Compute metrics
            yvpred = model_wrapper.get_predictions(version_).values
            if not model_wrapper.predict_order_books:
                price_mae = model_wrapper.price_mae(Yv, yvpred)
                price_dae = model_wrapper.price_dae(Yv, yvpred)
                price_rmae = model_wrapper.price_rmae(Yv, yvpred)
                price_smape = model_wrapper.price_smape(Yv, yvpred)        
                price_acc = model_wrapper.price_ACC(Yv, yvpred)
            else:
                price_mae = np.nan
                price_smape = np.nan
                price_acc = np.nan
                
            if model_wrapper.gamma > 0:
                OB_smape = model_wrapper.OB_smape(Yv, yvpred)
                OB_rsmape = model_wrapper.OB_rsmape(Yv, yvpred)        
                OB_acc = model_wrapper.OB_ACC(Yv, yvpred)                
                OB_racc = model_wrapper.OB_rACC(Yv, yvpred)
            else:
                OB_smape = np.nan
                OB_rsmape = np.nan
                OB_acc = np.nan
                OB_racc = np.nan
                
            ###### Store results            
            res = pandas.DataFrame({
                "country" : country,
                "ID" : ID,
                "OBs" : OBs,
                "version" : version,
                "val_price_mae" : price_mae,
                "val_price_dae" : price_dae,
                "val_price_rmae" : price_rmae,                
                "val_price_smape" : price_smape,            
                "val_price_ACC" : price_acc,
                "val_OB_rsmape" : OB_rsmape,                        
                "val_OB_rACC" : OB_racc,
                "val_OB_smape" : OB_smape,                        
                "val_OB_ACC" : OB_acc,
            }, index = [j*n + i])
            results = pandas.concat([results, res], ignore_index=True)

            ###### Store predictions and labels
            if not model_wrapper.predict_order_books:
                predicted_prices[j, i] = yvpred[:, model_wrapper.y_indices]
                real_prices[j, i] = Yv[:, model_wrapper.y_indices]
                
            if model_wrapper.gamma > 0:
                predicted_OB[j, i] = yvpred[:, model_wrapper.yOB_indices]
                real_OB[j, i] = Yv[:, model_wrapper.yOB_indices]

    return predicted_prices, real_prices, predicted_OB, real_OB, results

def normalize_shap(data):
    # Total contribution for each predicted label = 100%
    data = np.abs(data)
    per_label = data.sum(axis=2)    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j, :] /= per_label[i, j]
            
    data *= 100            
    return data

def retrieve_shap_values(IDs, country, dataset, OBs, N_VAL, N_SAMPLES,
                         N_GROUPS, folder,nh = 24, version=None):
    """
    Given a list of IDs, a fixed country, dataset and OBs, retrieve
    and process the shap values.
    """
    n = len(IDs)
    matrix = np.zeros((n , N_GROUPS))
    for i, ID in enumerate(IDs):

        ###### Create Model wrapper
        spliter = MySpliter(N_VAL, shuffle=False)        
        model_wrapper = OBNWrapper(
            "RESULTS", dataset, spliter=spliter, country=country,
            skip_connection=True, use_order_books=False,
            order_book_size=OBs, IDn=ID, tboard=folder)
        if version is None:
            version_ = model_wrapper.highest_version
        else:
            version_ = f"version_{version}"

        # Load raw shaps    
        shaps = np.load(model_wrapper.test_recalibrated_shape_path(version_))
        
        # Keep only contributions towards the prices
        shaps = shaps[model_wrapper.y_indices, :, :]

        # Keep only contributions from DATA (remove OB)
        shaps = shaps[:, :, model_wrapper.x_indices]

        # Normalize shaps : sum of shaps must be 100 for all labels/samples
        shaps = normalize_shap(shaps)
        
        # Average across all labels
        shaps = shaps.mean(axis=0)        
        
        # Group by variables
        N_GROUPS = len(model_wrapper.variables)
        grouped_shaps = np.zeros((N_SAMPLES, N_GROUPS))
        for j, variable in enumerate(model_wrapper.variables):
            inds_variable = model_wrapper.get_variable_indices(variable)
            grouped_shaps[:, j] = shaps[:, inds_variable].sum(axis=1)
            
        # Average across all samples
        grouped_shaps = grouped_shaps.mean(axis=0)
        
        # Store in the Matrix
        matrix[i, :] = grouped_shaps
        
    return matrix


def compute_dm_tests(countries, datasets, IDs, OBs,
                     predicted_prices, real_prices, predicted_OB, real_OB):
    nc = len(countries)
    n = len(IDs)    
    prices_pvalues = np.ones((nc, n, n)) * np.nan
    OB_pvalues = np.ones((nc, n, n))  * np.nan

    for i, (country, dataset) in enumerate(zip(countries, datasets)):
        for j, ID1 in enumerate(IDs):
            model_wrapper_1 = OBNWrapper(
                "RESULTS", dataset, country=country, IDn=ID1, tboard="RESULTS",
                skip_connection=True, use_order_books=False, order_book_size=OBs)
        
            for k, ID2 in enumerate(IDs):
                model_wrapper_2 = OBNWrapper(
                    "RESULTS", dataset, country=country, IDn=ID2, tboard="RESULTS",
                    skip_connection=True, use_order_books=False,order_book_size=OBs)

                # Compute the DM test on the Orderbooks
                if (model_wrapper_1.gamma > 0) and (model_wrapper_2.gamma > 0):
                    Y = real_OB[i, j]
                    Yhat1 = predicted_OB[i, j]
                    Yhat2 = predicted_OB[i, k]
                    if ID1 == ID2:
                        OB_pvalue = 1
                    else:
                        OB_pvalue = DM(Y, Yhat1, Yhat2, norm="smape")
                        
                    OB_pvalues[i, j, k] = OB_pvalue        

                # Compute the DM test on the Prices
                if ((not model_wrapper_1.predict_order_books)
                    and (not model_wrapper_2.predict_order_books)):
                    Y = real_prices[i, j]
                    Yhat1 = predicted_prices[i, j]
                    Yhat2 = predicted_prices[i, k]                
                    if ID1 == ID2:
                        prices_pvalue = 1
                    else:
                        prices_pvalue = DM(Y, Yhat1, Yhat2, norm="mae")
                    
                    prices_pvalues[i, j, k] = prices_pvalue
    return prices_pvalues, OB_pvalues

def compute_DM_tests_OBs(countries,datasets,IDs,OB_sizes,predicted_prices,
                         real_prices, predicted_OB, real_OB):
    prices_pvalues = {}
    OB_pvalues = {}
    for OBs in OB_sizes:
        key = str(OBs)
        pp, OBp = compute_dm_tests(
            countries,datasets,IDs,OBs,predicted_prices[key], real_prices[key],
            predicted_OB[key], real_OB[key])
        prices_pvalues[key] = pp
        OB_pvalues[key] = OBp
        
    return prices_pvalues, OB_pvalues

def compute_moments_seeds(res):
    temp = res.set_index("seed")
    means = temp.values.mean(axis=0)
    stds = temp.values.std(axis=0)
    temp.loc["mean", :] = means
    temp.loc["stds", :] = stds
    return temp

def merge_OB_tables(countries, datasets, IDs, OB_sizes, results):
    pass

def compute_DM_tests_2_OBs(countries,datasets,IDs, OBs1, OBs2, predicted_prices,
                           real_prices, predicted_OB, real_OB):
    price_tables = {}
    key1 = str(OBs1)
    key2 = str(OBs2)
    col = f"{key1} > {key2}"
    for i, (country, dataset) in enumerate(zip(countries, datasets)):
        price_tables[country] = pandas.DataFrame(
            columns=[col], index=IDs)        
        for j, ID1 in enumerate(IDs):
            model_wrapper_1 = OBNWrapper(
                "RESULTS", dataset, country=country, IDn=ID1, tboard="RESULTS",
                skip_connection=True, use_order_books=False, order_book_size=OBs1)
            model_wrapper_2 = OBNWrapper(
                "RESULTS", dataset, country=country, IDn=ID1, tboard="RESULTS",
                skip_connection=True, use_order_books=False, order_book_size=OBs2)

            # Compute the DM test on the Prices
            if ((not model_wrapper_1.predict_order_books)
                and (not model_wrapper_2.predict_order_books)):
                Y = real_prices[key1][i, j]
                Yhat1 = predicted_prices[key1][i, j]
                Yhat2 = predicted_prices[key2][i, j]                                
                price_tables[country].loc[ID1, col] = DM(Y,Yhat1,Yhat2, norm="mae")
    return price_tables            

def plot_DM_tests(pvalues, params, countries=[], IDs=[], label=""):    
    fig, axes = plt.subplots(2, 2, figsize=(1.9 * 19.2/4, 10.8),
                             sharex="col", sharey="row",
                             gridspec_kw={"hspace" : 0.15, "wspace" : 0.0})
    axes = axes.flatten()

    for i in range(4):
        has_data = True
        try:
            country = countries[i]
            ps = pvalues[i]
        except:
            has_data = False
            country = ""
        
        ax = axes[i]
        
        #### Remove nan columns
        mask = np.array([not np.isnan(ps[i]).all() for i in range(ps.shape[0])])
        ps = ps[mask, :][:, mask]

        tick_labels = ["$\mbox{DNN}_{Y}$", "$\mbox{DNN}_{OB}$", "$\mbox{DO}$",
                       "$\mbox{DO} + \mbox{DNN}_{OB}$",
                       "$\mbox{DO} + \mbox{DNN}_{Y}$",
                       "$\mbox{DO} + \mbox{DNN}_{Y, OB}$"]
        yy = -0.5
        
        #### Display the pvalues
        if has_data:
            im = ax.imshow(ps, cmap=cmap(), vmin=0, vmax=0.05, origin="lower")

            ##### Format the plot
            ax.set_title(country, fontsize=params["fontsize"], y=0.975)

            # X ticks
            ax.set_xticks(range(len(IDs[mask])))
            ax.set_xticklabels([])
            
            # Y ticks
            ax.set_yticks(range(len(IDs[mask])))
            ax.set_yticklabels(tick_labels, fontsize=params["fontsize_labels"])
            
            # Crosses on the diagonal
            ax.plot(range(len(IDs[mask])), range(len(IDs[mask])), 'wx')

    xx = 0.2
    for i in range(len(IDs[mask])):
        axes[2].text(i+xx, yy, tick_labels[i], rotation = 45, va="top",
                ha="right", fontsize = params["fontsize_labels"])
        axes[3].text(i+xx, yy, tick_labels[i], rotation = 45, va="top",
                ha="right", fontsize = params["fontsize_labels"])
        

    #axes[0].set_ylabel("Model ID", fontsize=params["fontsize_labels"])
    #axes[2].set_ylabel("Model ID", fontsize=params["fontsize_labels"])    
    #axes[2].set_xlabel("Model ID", fontsize=params["fontsize_labels"])
    #axes[3].set_xlabel("Model ID", fontsize=params["fontsize_labels"])    
    
    # Display the colorbar
    cbar = plt.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05,
                        pad=0.151)
    cbar.ax.set_xlabel("pvalue", fontsize=params["fontsize_labels"]) 
    cbar.ax.tick_params(labelsize=params["fontsize_labels"])   
    plt.suptitle(f"DM tests on the price forecasting task",
                 fontsize=params["fontsize"], y=0.95)
    

def plot_betas(res, IDs, OBs,country, dataset,params,ax_=None,col="val_price_smape",
               f=1, saw_tooth=[], ax2=None, div=None, title=None):
    if ax_ is None:
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
    else:
        ax = ax_
    
    temp = res.set_index("ID")
    betas = retrieve_betas(IDs, OBs, country, dataset, "", 365)       
        
    inds = np.argsort(betas)
    betas_plot = np.array(betas)[inds]
    error_plot = temp.loc[IDs, col].values[inds]
    
    not_keep_inds = [i for i, b in enumerate(betas_plot) if b in saw_tooth]
    keep_inds = [i for i, b in enumerate(betas_plot) if b not in saw_tooth]    

    betas_flat = betas_plot.copy()
    error_flat = error_plot.copy()
    
    for i in not_keep_inds:
        error_flat[i] = error_flat[keep_inds][i-f:i+f].mean()
    
    ax.plot(betas_flat, error_flat, c="g", linewidth=3)
    ax.grid("on")

    # X AXIS
    ax.set_xlim([betas[0], betas[-1]])

    beta_ticks = [b for i, b in enumerate(betas_plot) if (i % 2) == 0]
    ax.set_xticks(beta_ticks)
    ax.set_xticklabels(beta_ticks, fontsize=params["label_fontsize"])
    ax.set_xlabel("$\\beta$", fontsize=params["label_fontsize"])

    # Y AXIS
    ymin = error_flat.min()
    ymax = error_flat.max()
    if ax2 is not None:
        ymin, ymax = ax2.get_ylim()
        ymin /= div
        ymax /= div
        ax.set_yticks(ax2.get_yticks()/div)
    ax.tick_params(axis="y", labelsize=params["label_fontsize"])
        
    ax.set_ylim([ymin, ymax])
    
    if title is None:
        title = f"{col} for different values of $\\beta$"
    if ax2 is None:
        ax.set_title(title, fontsize=params["fontsize"], y=0.8)

    if ax_ is None:
        plt.show()

def paper_plot_all_betas(res, IDs, OBs, country, dataset, params):
    fig, axes = plt.subplots(3, 1, figsize=(19.2/2, 10.8), sharex="col",
                             gridspec_kw={"hspace" : 0.0})
    axes = axes.flatten()
    cols = ["val_price_mae", "val_price_dae", "val_price_smape", "val_price_rmae"]
    saw_tooth = [0.2, 0.45, 0.5]
    f = 3
    fontsize = params["fontsize"]

    titles = ["MAE (\\euro{}/MWh), RMAE", "DAE (\\euro{}/MWh)", "SMAPE (\%)"]
    res = copy.deepcopy(res)
    for i, col in enumerate(cols[:3]):
        plot_betas(res, IDs, OBs,country,dataset, params, ax_=axes[i], col=col, f=f,
                   saw_tooth=saw_tooth, title=titles[i])
    plt.suptitle("Evolution of the metrics while varying $\\beta$",
                 fontsize=params["fontsize"], y=0.92)

def plot_all_betas_1(res, IDs, country, dataset, fontsize=20, f=1, saw_tooth=[]):
    fig, axes = plt.subplots(3, 2, figsize=(19.2, 10.8), sharex="col")
    axes = axes.flatten()
    cols = ["val_price_mae", "val_price_dae", "val_price_smape",
            "val_price_rmae", "val_price_ACC"]

    axes[0].text(0.5, 1.1, "$\\alpha = 1 - \\beta$, $\\gamma = 0$",
                   transform=axes[0].transAxes, ha="center", fontsize=fontsize)
    for i, col in enumerate(cols):
        plot_betas(res, IDs, country, dataset, ax_=axes[i], col=col, f=f,
                   saw_tooth=saw_tooth)
    plt.show()          
    
def plot_all_betas(res, IDs1, IDs2, country, dataset, fontsize=20):
    fig, axes = plt.subplots(3, 2, figsize=(19.2, 10.8), sharex="col", sharey="row")
    cols = ["val_price_mae", "val_price_smape", "val_price_ACC"]

    axes[0,0].text(0.5, 1.1, "$\\alpha = 1 - \\beta$, $\\gamma = 0$",
                   transform=axes[0,0].transAxes, ha="center", fontsize=fontsize)
    axes[0,1].text(0.5, 1.1, "$\\alpha = \\gamma = \\frac{1 - \\beta}{2}$",
                   transform=axes[0,1].transAxes, ha="center", fontsize=fontsize)
    for i, ax in enumerate(axes[:, 0]):
        plot_betas(res,IDs1, country, dataset, ax_=ax, col=cols[i])
    for i, ax in enumerate(axes[:, 1]):
            plot_betas(res, IDs2, country, dataset, ax_=ax, col=cols[i])
    plt.show()    

def plot_shap_values_beta(matrix, IDs, N_VAL, country, dataset, OBs, folder,
                          params):
    """
    Plot the contributions of listed models IDs against the beta parameter
    """
    #### Retrieve betas
    betas = retrieve_betas(IDs, OBs, country, dataset, folder, N_VAL)
    
    #### Sort by betas
    indsort = np.argsort(betas)
    betas = betas[indsort]
    sorted_values = matrix[indsort, :]

    #### Compute difference with baseline
    baseline = sorted_values[0, :]
    matrix = (sorted_values - baseline)[1:]

    #### Display the matrix
    fig, ax = plt.subplots(1, figsize=(19.2, 10.8))
    vabs = np.abs(matrix).max()
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vabs, vmax=vabs)
    
    #### Arrange the plot
    ## Y Axis
    ax.set_ylabel("$\\beta$", fontsize=params["fontsize"])

    yticks_positions = np.array([i for i in range(len(betas[1:]))])[::-1]
    ax.set_yticks(yticks_positions)
    ax.set_yticklabels(betas[1:], fontsize=params["fontsize_labels"])

    plt.annotate("", xy=[-0.001, 0], xycoords="axes fraction",
                 xytext = [-0.001, 1.01], textcoords="axes fraction",
                 arrowprops={"arrowstyle": "<-", "linewidth" :5, "color" : "k"})    

    ## X Axis
    model_wrapper =  OBNWrapper(
        "RESULTS", dataset, spliter=MySpliter(N_VAL, shuffle=False),
        country=country, skip_connection=True, use_order_books=False,
        order_book_size=OBs, IDn=7, tboard=folder)
    xticks_labels = [model_wrapper.map_variable(mwv)
                     for mwv in model_wrapper.variables]    
    xticks_positions = np.array([i + 0.5 for i in range(len(xticks_labels))])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ypos = len(betas) - 1.6
    for xpos, text in zip(xticks_positions, xticks_labels):
        ax.text(xpos, ypos, text, fontsize=params["fontsize_labels"], rotation=45,
                va = "top", ha = "right")

    ## The colorbar
    cbar = plt.colorbar(im, ax=ax, location="top", shrink=0.3, pad=0.01)
    cbar.ax.tick_params(labelsize=params["fontsize_labels"], pad=-15)
    cbar.ax.tick_params(direction="in")    

    ## Title
    plt.suptitle(
        "Difference of contribution between $\\beta = 0$ and $\\beta > 0 \; (\%)$",
        fontsize=params["fontsize"], y=0.82)    

class MyFormatter(Formatter):
    def __call__(self, x, pos=None):
        x = mdates.num2date(x)  # Convert number to datetime
        if x.hour == 0:
            return x.strftime('%d/%m')  # return date
        else:
            return x.strftime('%Hh')  # return hour

def retrieve_betas(IDs, OBs, country, dataset, folder, N_VAL):
    betas = []    
    for i, ID in enumerate(IDs):
        
        ###### Create Model wrapper
        spliter = MySpliter(N_VAL, shuffle=False)        
        model_wrapper = OBNWrapper(
            "RESULTS", dataset, spliter=spliter, country=country,
            skip_connection=True, use_order_books=False,
            order_book_size=OBs, IDn=ID, tboard=folder)
        betas.append(model_wrapper.beta)
    betas = np.array(betas)
    return betas
        
def plot_predictions(predictions, real_prices, IDs, OBs, N_VAL, country, dataset,
                     folder, params, betas_to_plot=[0, 0.5, 1]):
    """
    Plot forecasts and real labels for several values of beta
    """
    fig, axes = plt.subplots(2, figsize=(19.2/2, 10.8), sharex=True,
                             gridspec_kw={"hspace": 0.0})
    ax = axes[0]
    
    #### Retrieve betas
    betas = retrieve_betas(IDs, OBs, country, dataset, folder, N_VAL)
    indsort = np.argsort(betas)
    betas = betas[indsort]    

    #### Retrieve test dates
    model_wrapper = OBNWrapper(
        "RESULTS", dataset, spliter=MySpliter(N_VAL, shuffle=False),country=country,
        skip_connection=True, use_order_books=False, order_book_size=OBs, IDn=7,
        tboard=folder)
        
    y = real_prices[0].reshape(-1)
    X, Y = model_wrapper.load_train_dataset()
    (Xtr, Ytr), (Xv, Yv) = model_wrapper.spliter(X, Y)
    _, test_dates = model_wrapper.spliter(model_wrapper.train_dates)
    xindices = [datetime.datetime(d.year,d.month,d.day)+datetime.timedelta(hours=h)
                for d in test_dates for h in range(24)]

    #### Filter betas
    inds = np.array([np.where(b == betas)[0][0] for b in betas_to_plot])
    
    #### Plot data
    cmap = plt.get_cmap("RdYlGn")
    for j, (i, beta) in enumerate(zip(inds, betas_to_plot)):
        #color_index = (j + 1) / (len(betas_to_plot) + 1)
        color_index = (j ) / (len(betas_to_plot))
        color = cmap(color_index)
        label = "$\\beta=" + str(beta) + "$"
        if beta == 0:
            color = "r"
        if beta == 1:
            color = "g"
        #pred_index = np.where(np.array(IDs) == IDs[i])[0][0]
        pred_index = i        
        ax.plot(
            xindices,predictions[pred_index].reshape(-1),
            label=label,c=color, linewidth=2.5)
    ax.plot(xindices, y, label="Real Prices", linewidth=4, color="k")

    #### Arrange plot
    ## x axis
    start = len(xindices) - 31 * 24 - 30 * 24 - 21 * 24 - 24
    stop = start + 72
    ax.set_xlim([xindices[start], xindices[stop]])

    ## y axis
    ax.set_ylabel("Forecasted Price (\euro{}/MWh)",fontsize=params["fontsize"],
                  labelpad=0.4)
    ax.set_ylim([0, 66])    

    ## Ticks
    ax.tick_params(which="both", axis="both", labelsize=params["fontsize_labels"])
    
    ## Grid
    ax.grid("on", axis="both", which="major")
    
    ## Legend
    ax.legend(fontsize=params["fontsize_labels"]*0.9, loc='upper left',
              bbox_to_anchor=(0.01, 1), framealpha=1)

    ## Title
    ax.set_title("Price forecasts and real price", fontsize=params["fontsize"])    
    
    #### Plot the variables
    axv = axes[1]

    ## Generation
    gen_inds = model_wrapper.get_variable_indices("Generation Forecasts")
    conso_inds, ren_inds = model_wrapper.get_variable_indices("Residual Load")    
        
    gen = 100 * (Xv[:, gen_inds] - Xv[:, gen_inds].mean(axis=0)) / Xv[:, gen_inds]
    axv.plot(xindices, gen.reshape(-1), label="Generation Forecasts", linewidth=2.5)

    ## residual Load
    rload = Xv[:, conso_inds] - Xv[:, ren_inds]
    rload = 100 * (rload - rload.mean(axis=0)) / rload
    axv.plot(xindices, rload.reshape(-1), label="Residual Load", linewidth=2.5)    

    ## General settings
    axv.set_ylim([-40, 22])
    axv.legend(fontsize=params["fontsize_labels"], loc='upper left',
               bbox_to_anchor=(0.01, 1), framealpha=1)
    axv.grid("on", axis="both", which="major")
    axv.set_title("Fundamental Variables", fontsize=params["fontsize"], y=0.01)

    ## Y axis
    axv.set_ylabel("Deviation (\%)", fontsize=params["fontsize"], labelpad=0.2)

    ## X axis
    axv.tick_params(which="both", axis="both", labelsize=params["fontsize_labels"])

    # Set locator : tick every 8 hours
    hours = mdates.HourLocator(interval=8)
    ax.xaxis.set_major_locator(hours)
    
    # Set formatter
    ax.xaxis.set_major_formatter(MyFormatter())

    # Set y locators
    ax.set_yticks(range(0, 70, 20))
    axv.set_yticks(range(-40, 30, 20))    
    

    # Plot vertical lines
    dates_to_plot = [
        datetime.datetime(2019, 10, 12, 2),
        datetime.datetime(2019, 10, 11, 23),
        datetime.datetime(2019, 10, 11, 1),
    datetime.datetime(2019, 10, 11, 4)]
    for date in dates_to_plot:
        for a in [ax, axv]:
            a.axvline(date, color="k", linestyle="--", linewidth=2)
    
def predict_order_book(IDs, OBs, N_VAL, country, dataset, dt, folder, params):
    """
    Load the models of the given IDs and predict order books
    """
    order_books = []

    #### Create a model_wrapper that will help
    model_wrapper = OBNWrapper(
        "RESULTS", dataset, spliter=MySpliter(N_VAL, shuffle=False),country=country,
        skip_connection=True, use_order_books=False, order_book_size=OBs, IDn=7,
        tboard=folder)
    
    #### Get real order books    
    X, Y = model_wrapper.load_train_dataset()
    (Xtr, Ytr), (Xv, Yv) = model_wrapper.spliter(X, Y)
    _, test_dates = model_wrapper.spliter(model_wrapper.train_dates)

    ind_date = np.where(np.array(test_dates) == dt.date())[0][0]
    h = dt.hour

    start = 0
    v_indices = np.array(
        [start + 3*model_wrapper.OBs*h+i for i in range(model_wrapper.OBs)])
    po_indices = np.array([v + model_wrapper.OBs for v in v_indices])
    p_indices = np.array([po + model_wrapper.OBs for po in po_indices])
    
    ## Load
    for ID in IDs:
        model_wrapper = OBNWrapper(
            "RESULTS", dataset, spliter=MySpliter(N_VAL, shuffle=False),
            country=country, skip_connection=True, use_order_books=False,
            order_book_size=OBs, IDn=ID, tboard=folder) 
        X, Y = model_wrapper.load_train_dataset()
        (Xtr, Ytr), (Xv, Yv) = model_wrapper.spliter(X, Y)   
        regr, version = model_wrapper.load(X, Y)
        
        Xv_scaled = regr.steps[0][1].transform(Xv)
        OBhat = regr.steps[1][1].predict_ob_scaled(Xv_scaled, regr)
        order_books.append(TorchOrderBook(OBhat))
        
    ## Load real OB
    model_wrapper = OBNWrapper(
        "RESULTS", dataset, spliter=MySpliter(N_VAL, shuffle=False),country=country,
        skip_connection=True, use_order_books=False, order_book_size=OBs, IDn=2,
        tboard=folder)
    X, Y = model_wrapper.load_train_dataset()
    (Xtr, Ytr), (Xv, Yv) = model_wrapper.spliter(X, Y)
    
    V = Yv[ind_date, v_indices]
    Po = Yv[ind_date, po_indices]
    P = Yv[ind_date, p_indices]
    
    real_OB = TorchOrderBook(torch.concatenate((
        torch.tensor(V).reshape(-1, 1),
        torch.tensor(Po).reshape(-1, 1),
        torch.tensor(P).reshape(-1, 1)), axis=1))

    return order_books, real_OB

def plot_predicted_order_books(
        IDs, OBs, N_VAL, country, dataset, dt, folder, params):
    """
    Plot forecasted Order books for the specified dates for the given IDs
    """
    fig, ax = plt.subplots(figsize=(19.2/2, 10.8), sharex=True,
                             gridspec_kw={"hspace": 0.1})    
    #### Retrieve betas
    betas = retrieve_betas(IDs, OBs, country, dataset, folder, N_VAL)
    indsort = np.argsort(betas)
    betas = betas[indsort]
    
    get_ploter(real_OB).display(ax_=ax)

    ax.set_ylim([-40, 60])    
    ax.set_xlim([2140, 2280])
    ax.set_title("Order Book of size {model_wrapper.OBs} for {dt} in Belgium")
    
    get_ploter(OB).display(ax_=ax, colors="b", labels="$\\beta = 1$")
    
def plot_shrinking(country, dataset, OBtries, dt, params):
    base_folder = os.environ["MOB"]
    data_folder = os.path.join(base_folder, "curves")

    df = load_real_prices(country)
    pandas_dt = pandas.Timestamp(dt)
    idt = np.where(np.array(df.index) == pandas_dt)[0][0]
    fig, axes = plt.subplots(1, 2, figsize=(19.2, 10.8),
                             gridspec_kw={"wspace" : 0.15})
    ax = axes[0]
    
    OBref = LoadedOrderBook(dt, os.path.join(data_folder, country))
    solverref = MinDual(OBref)
    pstar = solverref.solve("dual_derivative_heaviside")
    ploterref = get_ploter(OBref)
    ploterref.display(ax_=ax, colors="k", labels="Original order book",
                      linewidth=3, fit_to_data=False,step=0.01,
                      fontsize =params["fontsize_labels"], axlabel=False)
    ploterref.display(ax_=axes[1], colors="k", labels="Original order book",
                      linewidth=3, fit_to_data=False,step=0.01,
                      fontsize =params["fontsize_labels"], axlabel=False)

    pmin1 = -500
    pmax1 = 3000
    vmin1 = 1690
    vmax1 = 3000

    pmin2 = -20
    pmax2 = 60
    vmin2 = 2150
    vmax2 = 2350

    vstar = 2214
    axes[0].plot([vmin1, vstar], [pstar, pstar], linestyle="--", linewidth=3, c="r")
    ax.text(vmin1, pstar + 35, "Y = 22\\euro{}/MWh", c="r",
            fontsize=0.7*params["fontsize"])
    ax.set_ylabel("Price (EUR/MWh)", fontsize= params["fontsize"])
    plt.text(0.5, 0.025, "Cumulated volume (MWh)", transform = fig.transFigure,
             fontsize= params["fontsize"], ha="center")
    
    colors = ["c"]
    N = 1000
    for j, OBs in enumerate(OBtries):
        obd = OrderBookDataset(country, data_folder, df.index, OBs,
                               requires_grad=False, real_prices=df)
        OBshrink = SimpleOrderBook(TorchOrderBook(obd[idt][0]).orders)        
        ploter = get_ploter(OBshrink)
        solverref = MinDual(OBshrink)
        pstar = solverref.solve("dual_derivative_heaviside")        
        ploter.display(ax_=ax, colors=colors[j], labels=f"Shrinked to {OBs} orders",
                       linewidth=3, fit_to_data=False, step=0.01,
                       fontsize =params["fontsize_labels"], axlabel=False)
        ploter.display(ax_=axes[1], colors=colors[j],
                       labels=f"Shrinked to {OBs} orders",
                       linewidth=3, fit_to_data=False,step=0.01,
                       fontsize =params["fontsize_labels"], axlabel=False)

    axes[1].plot([vmin2, vstar], [pstar, pstar], linestyle="--", linewidth=3, c="r")
    axes[1].text(vmin2, pstar-4, "Y = 22\\euro{}/MWh", c="r",
                 fontsize=0.7 * params["fontsize"])
    
    ax.legend(fontsize=params["fontsize_labels"])    

    axes[0].tick_params(axis="both", labelsize=params["fontsize_labels"])
    axes[1].tick_params(axis="both", labelsize=params["fontsize_labels"])

    axes[0].set_ylim([pmin1, pmax1])
    axes[1].set_ylim([pmin2, pmax2])
    
    axes[0].set_xlim([vmin1, vmax1])
    axes[1].set_xlim([vmin2, vmax2])

    datetime_str = datetime.datetime.strftime(dt, "%Y-%m-%d, %Ham")    
    plt.suptitle(f"Belgian Order Book for the auction of {datetime_str}",
                 fontsize=params["fontsize"],  y=0.95)

        
    

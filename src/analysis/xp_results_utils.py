import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import numpy as np, pandas

from src.models.spliter import MySpliter
from src.models.torch_wrapper import OBNWrapper
from src.analysis.evaluate import DM

def cmap():
    red = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0.5, 50)[1:], [0]])
    green = np.concatenate([np.linspace(0.5, 1, 50), np.zeros(50)])
    blue = np.zeros(100)
    rgb_color_map = np.concatenate([red.reshape(-1, 1), green.reshape(-1, 1), 
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
                price_smape = model_wrapper.price_smape(Yv, yvpred)        
                price_acc = model_wrapper.price_ACC(Yv, yvpred)
            else:
                price_mae = np.nan
                price_smape = np.nan
                price_acc = np.nan
                
            if model_wrapper.gamma > 0:
                OB_smape = model_wrapper.OB_smape(Yv, yvpred)        
                OB_acc = model_wrapper.OB_ACC(Yv, yvpred)
            else:
                OB_smape = np.nan
                OB_acc = np.nan
                
            ###### Store results            
            res = pandas.DataFrame({
                "country" : country,
                "ID" : ID,
                "OBs" : OBs,
                "version" : version,
                "val_price_mae" : price_mae,
                "val_price_smape" : price_smape,            
                "val_price_ACC" : price_acc,
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

def plot_DM_tests(pvalues, countries=[], IDs=[], label="",
                  labels_fontsize=15):    
    fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.8))
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
        
        #### Display the pvalues
        if has_data:
            im = ax.imshow(ps, cmap=cmap(), vmin=0, vmax=0.05)

            ##### Format the plot
            ax.set_title(country)

            # X ticks
            ax.set_xticks(range(len(IDs[mask])))
            ax.set_xticklabels(IDs[mask], fontsize=labels_fontsize)
            ax.set_xlabel("Model ID")
        
            # Y ticks
            ax.set_yticks(range(len(IDs[mask])))
            ax.set_yticklabels(IDs[mask], fontsize=labels_fontsize)
            ax.set_ylabel("Model ID")        

            # Crosses on the diagonal
            ax.plot(range(len(IDs[mask])), range(len(IDs[mask])), 'wx')

    # Display the colorbar
    cbar = plt.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05,
                        pad=0.21)
    cbar.ax.set_xlabel("Pvalue of the DM test")            
    plt.suptitle(f"PValues of the {label}")    
    

def plot_betas(res, IDs, country, dataset, ax_=None, col="val_price_smape"):
    if ax_ is None:
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
    else:
        ax = ax_
    
    temp = res.set_index("ID")
    betas = []
    for ID in IDs:
        model_wrapper = OBNWrapper("RESULTS", dataset, country=country, IDn=ID)
        betas.append(model_wrapper.beta)        

    print(betas)
    inds = np.argsort(betas)
    ax.plot(np.array(betas)[inds], temp.loc[IDs, col].values[inds])
    ax.grid("on")

    # X AXIS
    ax.set_xticks(betas)
    ax.set_xticklabels(betas)
    ax.set_xlabel("$\\beta$")

    # Y AXIS
    #ax.set_yticks()
    #ax.set_xticklabels()
    ax.set_ylabel(col)
    ax.set_title(f"{col} for different values of $\\beta$", y=0.9)

    if ax_ is None:
        plt.show()

def plot_all_betas_1(res, IDs, country, dataset, fontsize=20):
    fig, axes = plt.subplots(3, 1, figsize=(19.2, 10.8), sharex="col", sharey="row")
    cols = ["val_price_mae", "val_price_smape", "val_price_ACC"]

    axes[0].text(0.5, 1.1, "$\\alpha = 1 - \\beta$, $\\gamma = 0$",
                   transform=axes[0].transAxes, ha="center", fontsize=fontsize)
    for i, ax in enumerate(axes):
        plot_betas(res, IDs, "FR", "Lyon", ax_=ax, col=cols[i])
    plt.show()          
    
def plot_all_betas(res, IDs1, IDs2, country, dataset, fontsize=20):
    fig, axes = plt.subplots(3, 2, figsize=(19.2, 10.8), sharex="col", sharey="row")
    cols = ["val_price_mae", "val_price_smape", "val_price_ACC"]

    axes[0,0].text(0.5, 1.1, "$\\alpha = 1 - \\beta$, $\\gamma = 0$",
                   transform=axes[0,0].transAxes, ha="center", fontsize=fontsize)
    axes[0,1].text(0.5, 1.1, "$\\alpha = \\gamma = \\frac{1 - \\beta}{2}$",
                   transform=axes[0,1].transAxes, ha="center", fontsize=fontsize)
    for i, ax in enumerate(axes[:, 0]):
        plot_betas(res,IDs1, "FR", "Lyon", ax_=ax, col=cols[i])
    for i, ax in enumerate(axes[:, 1]):
            plot_betas(res, IDs2, "FR", "Lyon", ax_=ax, col=cols[i])
    plt.show()    

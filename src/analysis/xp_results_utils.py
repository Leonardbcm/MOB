import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import numpy as np, pandas

from src.models.spliter import MySpliter
from src.models.torch_wrapper import OBNWrapper

def cmap():
    red = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0.5, 50)[1:], [0]])
    green = np.concatenate([np.linspace(0.5, 1, 50), np.zeros(50)])
    blue = np.zeros(100)
    rgb_color_map = np.concatenate([red.reshape(-1, 1), green.reshape(-1, 1), 
                                    blue.reshape(-1, 1)], axis=1)
    rgb_color_map = mpl.colors.ListedColormap(rgb_color_map)
    return rgb_color_map

def retrieve_results(IDs, countries, datasets, OBs, N_VAL, N_SAMPLES, nh = 24):
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
                order_book_size=OBs, IDn=ID, tboard="RESULTS")
            version = model_wrapper.highest_version
            stopped_epoch = model_wrapper.get_stopped_epoch(version)

            ###### Load DATA        
            X, Y = model_wrapper.load_train_dataset()
            X = X[:N_SAMPLES, :]
            Y = Y[:N_SAMPLES, :]
            (_, _), (Xv, Yv) = model_wrapper.spliter(X, Y)
            
            ###### Compute metrics
            yvpred = model_wrapper.get_predictions(version).values
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
    

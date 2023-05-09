import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import numpy as np, pandas

def cmap():
    red = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0.5, 50)[1:], [0]])
    green = np.concatenate([np.linspace(0.5, 1, 50), np.zeros(50)])
    blue = np.zeros(100)
    rgb_color_map = np.concatenate([red.reshape(-1, 1), green.reshape(-1, 1), 
                                    blue.reshape(-1, 1)], axis=1)
    rgb_color_map = mpl.colors.ListedColormap(rgb_color_map)
    return rgb_color_map

def plot_DM_tests(pvalues, countries=[], IDs=[], label="", labels_fontsize=15):    
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
        
        #### Display the pvalues
        if has_data:
            im = ax.imshow(ps, cmap=cmap(), vmin=0, vmax=0.1)

        ##### Format the plot
        ax.set_title(country)

        # X ticks
        ax.set_xticks(range(len(IDs)))
        ax.set_xticklabels(IDs, fontsize=labels_fontsize)
        ax.set_xlabel("Model ID")

        # Y ticks
        ax.set_yticks(range(len(IDs)))
        ax.set_yticklabels(IDs, fontsize=labels_fontsize)
        ax.set_ylabel("Model ID")        

        # Crosses on the diagonal
        ax.plot(range(len(IDs)), range(len(IDs)), 'wx')

    # Display the colorbar
    cbar = plt.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05,
                        pad=0.21)
    cbar.ax.set_xlabel("Pvalue of the DM test")            
    plt.suptitle(f"PValues of the {label}")    
    

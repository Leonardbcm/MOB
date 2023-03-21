%load aimport

import matplotlib

from sklearn.metrics import mean_absolute_error
from src.euphemia.orders import *
from src.euphemia.order_books import *
from src.euphemia.solvers import *
from src.euphemia.ploters import *
from src.models.torch_wrapper import OBNWrapper
from src.models.torch_models.weight_initializers import *
from src.models.parallel_scikit import set_all_seeds

################### Load data and configure default parameters
model_wrapper = OBNWrapper("TEST", "Lyon")
X, Y = model_wrapper.load_train_dataset()
ptemp = model_wrapper.params()
ptemp["early_stopping"] = None
ptemp["n_epochs"] = 500
ptemp["tensorboard"] = "OBN_scalers"
################### Configuration for the tests
# No Order Book scaling, only apply a label transformation
p11 = {
    "store_val_OBhat" : os.path.join(model_wrapper.folder(), "Forecasts", "P11"),
    "transformer" : "",
    "scale" : "",
    "weight_initializers" : []
}
p12 = {
    "store_val_OBhat" : os.path.join(model_wrapper.folder(), "Forecasts", "P12"),
    "transformer" : "Standard",
    "scale" : "",
    "weight_initializers" : []
}

# Apply an order book scaling with respect to the labels scaling
p21 = {
    "store_val_OBhat" : os.path.join(model_wrapper.folder(), "Forecasts", "P21"),
    "transformer" : "",
    "scale" : "MinMax",
    "weight_initializers" : []
}
p22 = {
    "store_val_OBhat" : os.path.join(model_wrapper.folder(), "Forecasts", "P22"),  
    "transformer" : "Standard",
    "scale" : "MinMax",
    "weight_initializers" : []
}

# Apply layer weight intialization to match the order books distribution
p31 = {
    "store_val_OBhat" : os.path.join(model_wrapper.folder(), "Forecasts", "P31"),
    "transformer" : "",
    "scale" : "Clip",
    "OB_weight_initializers" : [BiasInitializer("normal", 30, 40, -500, 3000), ],
}

# Layer weight init should also match the label transformation!
p32 = {
    "store_val_OBhat" : os.path.join(model_wrapper.folder(), "Forecasts", "P32"),
    "transformer" : "Standard",
    "scale" : "Clip",
    "OB_weight_initializers" : [BiasInitializer("normal", 30, 40, -500, 3000), ],
}

# Apply the sign coercion for the volumes on top of weight init
p41 = {
    "store_val_OBhat" : os.path.join(model_wrapper.folder(), "Forecasts", "P41"),
    "transformer" : "",
    "scale" : "Clip-Sign",
    "OB_weight_initializers" : [BiasInitializer("normal", 30, 40, -500, 3000), ],
}

# Layer weight init should also match the label transformation!
p42 = {
    "store_val_OBhat" : os.path.join(model_wrapper.folder(), "Forecasts", "P42"),
    "transformer" : "Standard",
    "scale" : "Clip-Sign",
    "OB_weight_initializers" : [BiasInitializer("normal", 30, 40, -500, 3000), ],
}
#################### Check XP settings
save_path = os.path.join(model_wrapper.folder(), "Forecasts")
params = copy.deepcopy(ptemp)
params["n_epochs"] = 1000
regrs = {}
for parameters, name in zip([p11, p12, p21, p22, p31, p32, p41, p42],
                            ["P11", "P12", "P21", "P22", "P31", "P32","P41","P42"]):
    param_temps = copy.deepcopy(params)
    param_temps.update(parameters)
    
    regr = model_wrapper.make(param_temps)
    set_all_seeds(0)
    regr.fit(X, Y)
    regrs[name] = regr
    (Xt, Yt), (Xv, Yv) = model_wrapper.spliter(X, Y)
    
    # Predict validation
    yhat = model_wrapper.predict_test(regr, Xv)
    OBhat = model_wrapper.predict_order_books(regr, Xv)
    
    np.save(os.path.join(save_path, name, "yvhat.npy"), yhat)
    np.save(os.path.join(save_path, name, "OBvhat.npy"), OBhat)

maes = {}
ploters = {}
for parameters, name in zip([p11, p12, p21, p22, p31, p32, p41, p42],
                            ["P11", "P12", "P21", "P22", "P31", "P32","P41","P42"]):
    param_temps = copy.deepcopy(params)
    param_temps.update(parameters)    
    
    # Forecasted OB for each epoch (train forecast used for the gradient)
    ploter = MultiplePlotter(
        model_wrapper.spliter, X, param_temps["n_epochs"],
        save_to_disk=param_temps["store_val_OBhat"],
        batch_size=param_temps["batch_size"], OBs=param_temps["OBs"])
    ploters[name] = ploter

    # Load and compute MAE
    yhat = np.load(os.path.join(save_path, name, "yvhat.npy"))
    maes[name] = mean_absolute_error(Yv, yhat)

########################## Plot initial price distribution for all orders
ploter = ExpPloter(ploters, save_path)
with matplotlib.rc_context({ "text.usetex" : True,
                             "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                             "font.family" : ""}):
    plt.close("all")    
    ploter.distribution("validation", linewidth=4, fontsize=20)
    plt.show()

########################### Plot all order books forecast for the given epochs
with matplotlib.rc_context({"text.usetex" : True,
                            "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                            "font.family" : ""}):
    plt.close("all")    
    ploters["P21"].display(20, 8, "validation",
                           epochs=[i for i in np.arange(0, 100, 10)] + [-1],
                           colormap="viridis", linewidth=4, fontsize=20)
    plt.show()

########################### Display price forecasts and the real ones
with matplotlib.rc_context({"text.usetex" : True,
                            "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                            "font.family" : ""}):
    plt.close("all")    
    ploter.price_forecasts(Yv)
    plt.show()

############################# Display distribution of the forecasted prices
with matplotlib.rc_context({"text.usetex" : True,
                            "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                            "font.family" : ""}):
    plt.close("all")    
    ploter.price_forecasts_distributions(Yv)
    plt.show()

############################# Relaunch good models without logging
params = copy.deepcopy(ptemp)
params["n_epochs"] = 1000
params["store_val_OBhat"] = False
ploters = {}
regrs = {}
for parameters, name in  zip([p11, p12, p21, p22, p31, p32, p41, p42],
                            ["P11", "P12", "P21", "P22", "P31", "P32","P41","P42"]):
    param_temps = copy.deepcopy(params)
    param_temps.update(parameters)
    
    regr = model_wrapper.make(param_temps)
    set_all_seeds(0)
    regr.fit(X, Y)
    regrs[name] = regr
    (Xt, Yt), (Xv, Yv) = model_wrapper.spliter(X, Y)
    
    # Predict validation
    yhat = model_wrapper.predict_test(regr, Xv)
    OBhat = model_wrapper.predict_order_books(regr, Xv)
    
    np.save(os.path.join(save_path, name, "full_yvhat.npy"), yhat)
    np.save(os.path.join(save_path, name, "full_OBvhat.npy"), OBhat)

maes = {}
for parameters, name in  zip([p11, p12, p21, p22, p31, p32, p41, p42],
                            ["P11", "P12", "P21", "P22", "P31", "P32","P41","P42"]):
    param_temps = copy.deepcopy(params)
    param_temps.update(parameters)    
    
    # Forecasted OB for each epoch (train forecast used for the gradient)
    ploter = MultiplePlotter(
        model_wrapper.spliter, X, param_temps["n_epochs"],
        save_to_disk=param_temps["store_val_OBhat"],
        batch_size=param_temps["batch_size"], OBs=param_temps["OBs"])
    ploters[name] = ploter

    # Load and compute MAE
    yhat = np.load(os.path.join(save_path, name, "full_yvhat.npy"))
    maes[name] = mean_absolute_error(Yv, yhat)
    
######################### In-depth analysis
# Layer weight init should also match the label transformation!
p32 = {
    "store_OBhat" : os.path.join(model_wrapper.folder(), "Forecasts", "P32"),
    "transformer" : "",
    "scale" : "Clip",
    "weight_initializers" : [BiasInitializer("normal", 30, 40, -500, 3000), ],
    "n_epochs" : 1,
}

param_temps = copy.deepcopy(ptemp)
param_temps.update(p32)

regr = model_wrapper.make(param_temps)
set_all_seeds(0)
regr.fit(X, Y)

ploter = MultiplePlotter(
    model_wrapper.spliter, X, param_temps["n_epochs"],
    save_to_disk=param_temps["store_OBhat"],
    batch_size=param_temps["batch_size"], OBs=param_temps["OBs"])

ploter.distribution("train", fontsize=12, epochs=[0])
res = ploter.get_extreme_orders("train")

ploter.plot_extreme_orders("train", steps=-1, epochs=0, variables=-1, d=0, h=3,
                           linewidth=2, label_fontsize=20, fontsize=30)

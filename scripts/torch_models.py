%load aimport

from sklearn.metrics import mean_absolute_error

from src.euphemia.orders import *
from src.euphemia.order_books import *
from src.euphemia.solvers import *
from src.euphemia.ploters import *

from src.models.spliter import MySpliter
from src.models.torch_wrapper import OBNWrapper
from src.models.torch_models.weight_initializers import *
import src.models.parallel_scikit as ps

############## Construct model wrapper and load data
spliter = MySpliter(365, shuffle=False)
model_wrapper = OBNWrapper("TEST", "Lyon", spliter=spliter)
X, Y = model_wrapper.load_train_dataset()

############## Set some params
ptemp = model_wrapper.params()
ptemp["transformer"] = ""
ptemp["NN1"] = (888, )
ptemp["OBN"] = ()
ptemp["OBs"] = 100
ptemp["early_stopping"] = None
ptemp["very_early_stopping"] = True
ptemp["n_epochs"] = 1
ptemp["tensorboard"] = "OBN"
ptemp["OB_weight_initializers"] = {
    "polayers" : [BiasInitializer("normal", 30, 40, -500, 3000)],
    "poplayers" : [BiasInitializer("normal", 30, 40, -500, 3000) ]}
regr = model_wrapper.make(ptemp)
(Xt, Yt), (Xv, Yv) = model_wrapper.spliter(X, Y)

############## Fit model and predict validation set
ps.set_all_seeds(0)
regr.fit(X, Y)
yhat = model_wrapper.predict_val(regr, Xv)
mean_absolute_error(Yv, yhat)

# Sample a configuration from the search space
search_space = model_wrapper.get_search_space(n=X.shape[0])
([ptemp], [seed]) = ps.get_param_list_and_seeds(
    search_space, 1, model_wrapper=model_wrapper)

# Fit the model and compute the error using this configuration
regr = model_wrapper.make(model_wrapper._params(ptemp))
(Xt, Yt), (Xv, Yv) = model_wrapper.spliter(X, Y)
ps.set_all_seeds(seed)
regr.fit(X, Y)
yhat = model_wrapper.predict_val(regr, Xv)
mean_absolute_error(Yv, yhat)

################## Use fitted model to forecast OB
# Forecasted OB for each epoch (train forecast used for the gradient)
OBhat = model_wrapper.predict_order_books(regr, Xv)
ploter = MultiplePlotter(regr, X)
ploter.display(0, 0, epochs=-1, colormap="viridis")

# Plot tensor distirbution across epochs
ploter.distribution(epochs=-1, steps=-1)

order_book = TorchOrderBook(OBhat[d, h])
solver = MinDual(order_book)
ploter = get_ploter(order_book, solver)
ploter.arrange_plot("display", "dual_function",
                    ["dual_derivative", {"method" : "sigmoid"}])

####################  LOAD A CONFIG AND CHECK PRICE DISTRIBUTION
spliter = MySpliter(365, shuffle=False)
model_wrapper = OBNWrapper("TEST", "Lyon", spliter=spliter)
X, Y = model_wrapper.load_train_dataset()

############## Set some params
ptemp = model_wrapper.params()
ptemp["transformer"] = ""
ptemp["NN1"] = (888, )
ptemp["OBN"] = ()
ptemp["OBs"] = 100
ptemp["early_stopping"] = None
ptemp["n_epochs"] = 1
ptemp["tensorboard"] = "OBN"
ptemp["OB_weight_initializers"] = {
    "polayers" : [BiasInitializer("normal", 30, 40, -500, 3000)],
    "poplayers" : [BiasInitializer("normal", 30, 40, -500, 3000) ]}
regr = model_wrapper.make(ptemp)
(Xt, Yt), (Xv, Yv) = model_wrapper.spliter(X, Y)

############## Fit model and predict validation set
ps.set_all_seeds(0)
regr.fit(X, Y)
yhat = model_wrapper.predict_val(regr, Xv)
mean_absolute_error(Yv, yhat)

# Sample a configuration from the search space
search_space = model_wrapper.get_search_space(n=X.shape[0])
([ptemp], [seed]) = ps.get_param_list_and_seeds(
    search_space, 1, model_wrapper=model_wrapper)

# Fit the model and compute the error using this configuration
regr = model_wrapper.make(model_wrapper._params(ptemp))
(Xt, Yt), (Xv, Yv) = model_wrapper.spliter(X, Y)
ps.set_all_seeds(seed)
regr.fit(X, Y)
yhat = model_wrapper.predict_val(regr, Xv)
mean_absolute_error(Yv, yhat)

################## Use fitted model to forecast OB
# Forecasted OB for each epoch (train forecast used for the gradient)
OBhat = model_wrapper.predict_order_books(regr, Xv)
ploter = MultiplePlotter(regr, X)
ploter.display(0, 0, epochs=-1, colormap="viridis")

# Plot tensor distirbution across epochs
ploter.distribution(epochs=-1, steps=-1)

order_book = TorchOrderBook(OBhat[d, h])
solver = MinDual(order_book)
ploter = get_ploter(order_book, solver)
ploter.arrange_plot("display", "dual_function",
                    ["dual_derivative", {"method" : "sigmoid"}])

####################  LOAD A CONFIG AND CHECK PRICE DISTRIBUTION
############## Construct model wrapper and load data
spliter = MySpliter(365, shuffle=False)
model_wrapper = OBNWrapper("OBN_TSCHORA", "Lyon", spliter=spliter)
X, Y = model_wrapper.load_train_dataset()

results = model_wrapper.load_results()
ptemp = dict(results.loc[0])
ptemp["n_epochs"] = 2
ptemp["store_val_OBhat"] = os.path.join(
    model_wrapper.folder(), "Forecasts", "params_1")
ptemp["store_OBhat"] = os.path.join(
    model_wrapper.folder(), "Forecasts", "params_1")
ptemp["batch_size"] = 10
params = model_wrapper.params()
params.update(ptemp)
regr = model_wrapper.make(params)
ps.set_all_seeds(0)
regr.fit(X, Y)

ploter = MultiplePlotter(
    model_wrapper.spliter, X, params["n_epochs"],
    save_to_disk=params["store_OBhat"],
    batch_size=params["batch_size"], OBs=params["OBs"])

ploter.distribution("train", fontsize=12, epochs=[0])
ploter.price_forecasts_distributions("train", 0)

ploter.scaling_summary(regr, Y)





################### Gaussian manipulation
n = 10000
nin = 50
nout = 100
x = np.random.normal(0, 1, size=(n, nin))

muW = 0
sW = 1
W = np.random.normal(muW, sW, size=(nin, nout))

mub = 0
sb = 1
b = np.random.normal(mub, sb, size=nout)
#b[0] = -500
#b[1] = 3000

xout = np.matmul(x, W) + b

plt.hist(x.reshape(-1), bins=100, histtype='step', label='input', density=True)
plt.hist(W.reshape(-1), bins=100, histtype='step', label='Weights', density=True)
plt.hist(b.reshape(-1), bins=10, histtype='step', label='Bias', density=True)
plt.hist(xout.reshape(-1), bins=100, histtype='step', label='output', density=True)
plt.legend()
plt.show()


################### Using uniform
n = 10000
nin = 50
nout = 100
x = np.random.normal(0, 1, size=(n, nin))

k = math.sqrt(1 / nin)
W = np.random.uniform(-k, k, size=(nin, nout))
b = np.random.uniform(-k, k, size=nout)

xout = np.matmul(x, W) + b
xtest = np.random.normal(0, 0.6, size=(n, nin))

import matplotlib
with matplotlib.rc_context({ "text.usetex" : True,
                             "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc} \usepackage{mathtools}",
                             "font.family" : ""}):
    plt.close("all")
    fontsize = 20
    fig, ax = plt.subplots(1)
    ax.hist(x.reshape(-1), bins=100, histtype='step', linewidth=4,
             label="$X^{L-1} \sim N(0, 1)$", density=True, edgecolor="b")
    ax.hist(xout.reshape(-1), bins=100, histtype='step', edgecolor="r",
             label="$\widehat{OB} = W^LX^{L-1} + b^L$", density=True, linewidth=4)
    ax.grid("on")
    ax.legend(fontsize=fontsize, bbox_to_anchor=(0, 0.5, 1, 0.5), ncols=2,
              mode="expand",  borderaxespad=0, bbox_transform=fig.transFigure)
    ax.tick_params(labelsize=fontsize)
    plt.show()


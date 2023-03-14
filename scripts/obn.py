%load aimport

from sklearn.metrics import mean_absolute_error
from src.euphemia.orders import *
from src.euphemia.order_books import *
from src.euphemia.solvers import *
from src.euphemia.ploters import *
from src.models.obn_wrapper import OBNWrapper
from src.models.weight_initializers import *
from src.models.parallel_scikit import set_all_seeds

model_wrapper = OBNWrapper("TEST", "Lyon", countries_to_predict="not_graph")
X, Y = model_wrapper.load_train_dataset()
ptemp = model_wrapper.params()
ptemp["transformer"] = ""
ptemp["NN1"] = (888, )
ptemp["OBN"] = (37, )
ptemp["OBs"] = 100
ptemp["early_stopping"] = None
ptemp["n_epochs"] = 5
ptemp["tensorboard"] = "OBN"
ptemp["store_losses"] = True
ptemp["store_OBhat"] = True
ptemp["scale"] = "Clip"
ptemp["k"] = 100
ptemp["niter"] = 30
ptemp["OB_weight_initializers"] = {
    "polayers" : [Initializer("normal", "weight", 1, 1),
                  BiasInitializer("normal", 30, 40, -500, 3000)],
    "poplayers" : [Initializer("normal", "weight", 1, 1),
                   BiasInitializer("normal", 30, 40, -500, 3000) ]}
regr = model_wrapper.make(ptemp)
set_all_seeds(0)
regr.fit(X, Y)
yhat = model_wrapper.predict_test(regr, X)
OBhat = model_wrapper.predict_order_books(regr, X)
mean_absolute_error(Y, yhat)

# Forecasted OB for each epoch (train forecast used for the gradient)
ploter = MultiplePlotter(regr, X)
ploter.display(0, 0, epochs=-1, colormap="viridis")

# Plot tensor distirbution across epochs
ploter.distribution(epochs=-1, steps=-1)

################## Use fitted model to forecast OB
order_book = TorchOrderBook(OBhat[d, h])
solver = MinDual(order_book)
ploter = get_ploter(order_book, solver)
ploter.arrange_plot("display", "dual_function",
                    ["dual_derivative", {"method" : "sigmoid"}])
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


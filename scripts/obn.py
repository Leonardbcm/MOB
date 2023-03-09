%load_ext autoreload
%autoreload 1
%aimport src.models.model_wrapper
%aimport src.models.obn_wrapper
%aimport src.models.obn.obn
%aimport src.models.obn.torch_obn
%aimport src.models.obn.torch_solver
%aimport src.models.scalers
%aimport src.models.callbacks

%aimport src.euphemia.orders
%aimport src.euphemia.order_books
%aimport src.euphemia.solvers
%aimport src.euphemia.ploters

from sklearn.metrics import mean_absolute_error
from src.euphemia.orders import *
from src.euphemia.order_books import *
from src.euphemia.solvers import *
from src.euphemia.ploters import *
from src.models.obn_wrapper import OBNWrapper
from src.models.parallel_scikit import set_all_seeds

model_wrapper = OBNWrapper("TEST", "Lyon", countries_to_predict="not_graph")
X, Y = model_wrapper.load_train_dataset()
ptemp = model_wrapper.params()
ptemp["transformer"] = ""
ptemp["NN1"] = (240, )
ptemp["OBN"] = (10, )
ptemp["OBs"] = 2
ptemp["n_epochs"] = 5
ptemp["tensorboard"] = "OBN"
ptemp["store_losses"] = True
ptemp["store_OBhat"] = True
ptemp["clip"] = False
ptemp["scale"] = False

ptemp["k"] = 100
ptemp["niter"] = 25
regr = model_wrapper.make(ptemp)

set_all_seeds(0)
regr.fit(X, Y)
yhat = model_wrapper.predict_test(regr, X)
OBhat = model_wrapper.predict_order_books(regr, X)
mean_absolute_error(Y, yhat)

# Forecasted OB for each epoch (train forecast used for the gradient)
ploter = MultiplePlotter(regr, X)
d = 0
h = 0
ploter.display(d, h, epochs=range(1, 3), colormap="viridis")

################## Use fitted model to forecast OB
order_book = TorchOrderBook(OBhat[d, h])
solver = MinDual(order_book)
ploter = get_ploter(order_book, solver)
ploter.arrange_plot("display", "dual_function",
                    ["dual_derivative", {"method" : "sigmoid"}])

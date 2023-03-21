%load aimport

from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

from src.models.spliter import MySpliter
from src.models.svr_wrapper import MultiSVRWrapper, ChainSVRWrapper
from src.models.parallel_scikit import set_all_seeds

############## Construct model wrapper and load data
spliter = MySpliter(365, shuffle=False)
model_wrapper = MultiSVRWrapper("TEST", "Lyon", spliter=spliter)
X, Y = model_wrapper.load_train_dataset()

############## Set some params
ptemp = model_wrapper.params()
ptemp['max_iter'] = 500
regr = model_wrapper.make(ptemp)

############## Fit model and predict validation set
(Xt, Yt), (Xv, Yv) = model_wrapper.spliter(X, Y)

set_all_seeds(0)
regr.fit(Xt, Yt)
yhat = model_wrapper.predict_val(regr, Xv)
mean_absolute_error(Yv, yhat)

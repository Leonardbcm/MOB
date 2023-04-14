%load aimport

from sklearn.metrics import mean_absolute_error

from src.models.spliter import MySpliter
from src.models.keras_wrapper import DNNWrapper, CNNWrapper
import src.models.parallel_scikit as ps

############### Use a DNN
# Instantiate wrapper
spliter = MySpliter(365, shuffle=False)
model_wrapper = DNNWrapper("TEST", "Lyon", spliter=spliter)
X, Y = model_wrapper.load_train_dataset()
(Xtr, Ytr), (Xv, Yv) = model_wrapper.spliter(X, Y)

# Use default params to isntantiate a model
ptemp = model_wrapper.params()
ptemp["n_epochs"] = 1000
ptemp["batch_size"] = 30
ptemp["neurons_per_layer"] = (888, 37)
ptemp["transformer"] = "Standard"
ptemp["scaler"] = "Standard"
ptemp["early_stopping"] = None
ptemp["dropout_rate"] = 0.1
regr = model_wrapper.make(ptemp)

# Fit the model and compute the error
ps.set_all_seeds(0)
regr.fit(X, Y)
yhat = model_wrapper.predict_val(regr, Xv)
mean_absolute_error(Yv, yhat)

# Compute test set error
Xt, Yt = model_wrapper.load_test_dataset()
ythat = model_wrapper.predict_val(regr, Xt)
mean_absolute_error(Yt, ythat)

# Sample a configuration from the search space
search_space = model_wrapper.get_search_space(n=X.shape[0])
([ptemp], [seed]) = ps.get_param_list_and_seeds(
    search_space, 1, model_wrapper=model_wrapper)

# Fit the model and compute the error using this configuration
regr = model_wrapper.make(model_wrapper._params(ptemp))
ps.set_all_seeds(seed)
regr.fit(X, Y)
yhat = model_wrapper.predict_val(regr, Xv)
mean_absolute_error(Yv, yhat)

############## Use a CNN
# Instantiate wrapper
spliter = MySpliter(365, shuffle=False)
model_wrapper = CNNWrapper("TEST", "Lyon", spliter=spliter)
X, Y = model_wrapper.load_train_dataset()

# Use default params to isntantiate a model
ptemp = model_wrapper.params()
regr = model_wrapper.make(ptemp)

# Fit the model and compute the error
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
ps.set_all_seeds(seed)
regr.fit(X, Y)
yhat = model_wrapper.predict_val(regr, Xv)
mean_absolute_error(Yv, yhat)

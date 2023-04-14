%load aimport

import itertools
from sklearn.metrics import mean_absolute_error

from src.euphemia.orders import *
from src.euphemia.order_books import *
from src.euphemia.solvers import *
from src.euphemia.ploters import *

from src.models.torch_models.ob_datasets import OrderBookDataset
from src.analysis.utils import load_real_prices
from src.models.spliter import MySpliter
from src.models.torch_wrapper import OBNWrapper
from src.models.torch_models.torch_obn import SolvingNetwork
from src.models.torch_models.weight_initializers import *
import src.models.parallel_scikit as ps
from src.analysis.evaluate import mae, smape

############## Construct model wrapper and load data
spliter = MySpliter(365, shuffle=False)
model_wrapper = OBNWrapper("TEST", "Lahaye", country="NL", spliter=spliter,
                           use_order_books=True, order_book_size=20,
                           separate_optim=True)
OB_features, OB_columns, OB_lab, OB_labels = model_wrapper.load_order_books()
OB_features.drop(datetime.date(2016, 1, 1), inplace=True)
OB_lab.drop(datetime.date(2016, 1, 1), inplace=True)
Yhat = OB_features.values
Y = OB_lab.values

############## Compute persistence error
model_wrapper.init_indices()

# Error on the volumes
Yv = Y[:, model_wrapper.v_indices].reshape(-1, model_wrapper.order_book_size)
Yhatv = Yhat[:, model_wrapper.v_indices].reshape(-1, model_wrapper.order_book_size)

maes_v = mae(Yv, Yhatv, mean=False)
smapes_v = smape(Yv, Yhatv, mean=False)

# Weighted smape using number of MWh
smapes = 200 * np.abs(Yv - Yhatv) / (0.001 + np.abs(Yv) + np.abs(Yhatv))
np.average(smapes, weights=np.abs(Yv) + np.abs(Yhatv))

# Error on Pos
Ypo = Y[:, model_wrapper.po_indices].reshape(-1, model_wrapper.order_book_size)
Yhatpo =Yhat[:, model_wrapper.po_indices].reshape(-1, model_wrapper.order_book_size)

maes_po = mae(Ypo, Yhatpo, mean=False)
smapes_po = smape(Ypo, Yhatpo, mean=False)

# Error on Ps
Yp = Y[:, model_wrapper.p_indices].reshape(-1, model_wrapper.order_book_size)
Yhatp =Yhat[:, model_wrapper.p_indices].reshape(-1, model_wrapper.order_book_size)

maes_p = mae(Yp, Yhatp, mean=False)
smapes_p = smape(Yp, Yhatp, mean=False)

############ Plot ob difference
idx = np.where(OB_features.index == pandas.to_datetime(datetime.date(2016, 10, 30)))[0][0] * 24 +2
fig, ax = plt.subplots(1)
OB = TorchOrderBook(np.concatenate(
    (Yv[idx].reshape(-1, 1),
     Ypo[idx].reshape(-1, 1),
     Yp[idx].reshape(-1, 1)), axis=1))
get_ploter(OB).display(ax_=ax, labels="Labels", colors="r")

OB = TorchOrderBook(np.concatenate(
    (Yhatv[idx].reshape(-1, 1),
     Yhatpo[idx].reshape(-1, 1),
     Yhatp[idx].reshape(-1, 1)), axis=1))
get_ploter(OB).display(ax_=ax, labels="Predictions", colors="b")
ax.legend(); plt.show()

############# Inspection
i = 10; plt.plot(Yv[:, i]); plt.show()

############# Compute dual difference
winter_to_summer = [datetime.datetime(2016, 3, 27, 2),
                    datetime.datetime(2017, 3, 26, 2),
                    datetime.datetime(2018, 3, 25, 2),
                    datetime.datetime(2019, 3, 31, 2),
                    datetime.datetime(2020, 3, 29, 2),
                    datetime.datetime(2021, 3, 28, 2)]
df = load_real_prices("FR")

dual_smapes = np.zeros(Yv.shape[0])
for idx in range(Yv.shape[0]):
    date_time = OB_lab.index[int(idx/24)]+datetime.timedelta(hours=idx % 24)
    if date_time in winter_to_summer:
        date_time -= datetime.timedelta(hours=24)
        
    ref_price = df.loc[date_time, "price"]
    OB = TorchOrderBook(np.concatenate(
        (Yv[idx].reshape(-1, 1),
         Ypo[idx].reshape(-1, 1),
         Yp[idx].reshape(-1, 1)), axis=1))
    true_dual = np.array([o.dual_function(ref_price) for o in OB.orders])
    
    OBhat = TorchOrderBook(np.concatenate(
        (Yhatv[idx].reshape(-1, 1),
         Yhatpo[idx].reshape(-1, 1),
         Yhatp[idx].reshape(-1, 1)), axis=1))
    hat_dual = np.array([o.dual_function(ref_price) for o in OBhat.orders])
    dual_smapes[idx] = 200 * np.abs(true_dual.sum() - hat_dual.sum()) / (true_dual.sum() + hat_dual.sum())



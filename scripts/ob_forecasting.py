%load aimport

import itertools

from src.euphemia.orders import *
from src.euphemia.order_books import *
from src.euphemia.solvers import *
from src.euphemia.ploters import *

from src.analysis.utils import load_real_prices
from src.analysis.evaluate import ACC, smape, relative_error
from src.analysis.ob_forecasting_utils import *

from src.models.spliter import MySpliter
from src.models.torch_wrapper import OBNWrapper

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

############# Compare Dual difference, OB difference and price difference 
country = "FR"
dataset = "Lyon"
OBs = 20

# Load Data
model_wrapper = OBNWrapper(
    "TEST", dataset, country=country, spliter=MySpliter(365, shuffle=False),
    use_order_books=True, order_book_size=OBs, separate_optim=True)

Vloss, Ploss, Poloss, dual_loss, price_loss = compute_losses(model_wrapper)
plot_losses(Vloss, Ploss, Poloss, dual_loss, price_loss, OBs)

############################# Compute Cor coef
losses = np.concatenate((Vloss, Vloss.mean(axis=1).reshape(-1, 1),
                         Ploss, Ploss.mean(axis=1).reshape(-1, 1),
                         Poloss, Poloss.mean(axis=1).reshape(-1, 1),
                         dual_loss.reshape(-1, 1),price_loss.reshape(-1, 1)),axis=1)
coefs = np.corrcoef(losses.transpose())
plot_corrcoef(Vloss, Ploss, Poloss, dual_loss, price_loss, OBs)

############################# Get coefs for all combination
countries = ["FR", "DE", "BE", "NL"]
datasets = ["Lyon", "Munich", "Bruges", "Lahaye"]
order_book_sizes = [20, 50, 100, 250]

mean_coefs = np.zeros((len(countries), len(order_book_sizes), 3))
for i, (country, dataset) in enumerate(zip(countries, datasets)):
    for j, OBs in enumerate(order_book_sizes):
        model_wrapper = OBNWrapper(
            "TEST", dataset, country=country, spliter=MySpliter(365, shuffle=False),
            use_order_books=True, order_book_size=OBs, separate_optim=True)

        Vloss, Ploss, Poloss, dual_loss, price_loss = compute_losses(
            model_wrapper, relative_error)        
        losses = np.concatenate(
            (Vloss, Vloss.mean(axis=1).reshape(-1, 1),
             Ploss, Ploss.mean(axis=1).reshape(-1, 1),
             Poloss, Poloss.mean(axis=1).reshape(-1, 1),
             dual_loss.reshape(-1, 1),price_loss.reshape(-1, 1)),axis=1)
    
        coefs = np.corrcoef(losses.transpose())
        mean_coefs[i, j] = coefs[[OBs, 2*(OBs)+1, 3*(OBs)+1], -1]

np.save(os.path.join(os.environ["MOB"], "coeffs.npy"), mean_coefs)

%load aimport

from src.models.torch_wrapper import OBNWrapper
from src.models.spliter import MySpliter
from src.models.scalers import OBNScaler

from src.models.scalers import *
from sklearn.preprocessing import StandardScaler

spliter = MySpliter(365, shuffle=False)
model_wrapper = OBNWrapper("TEST", "Bruges", country="BE", spliter=spliter,
                           skip_connection=False,  use_order_books=False,
                           order_book_size=20, alpha=1, beta=0, gamma=0)
X, Y = model_wrapper.load_train_dataset()

ob_scaler = OBNScaler(
    "Standard", "Standard", model_wrapper.order_book_size,
    model_wrapper.past_price_col_indices(), model_wrapper.N_INPUT,
    model_wrapper.N_DATA, model_wrapper.x_indices, model_wrapper.v_indices,
    model_wrapper.po_indices, model_wrapper.p_indices, spliter=spliter)

ob_scaler.fit(X)
Xt = ob_scaler.transform(X)
Xr = ob_scaler.inverse_transform(Xt)
np.abs(X - Xr).mean()

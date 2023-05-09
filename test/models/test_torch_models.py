import pytest, os

from src.models.spliter import MySpliter
from src.models.torch_wrapper import OBNWrapper

####### MAIN PARAMS
N_EPOCHS = 1
N_SAMPLES = 10
N_VAL = 4
BATCH = 2

class TestTorchModel():
    def get_model(self, country, dataset, OBs, IDn):
        spliter = MySpliter(N_VAL, shuffle=False)
        model_wrapper = OBNWrapper("TEST", dataset, country=country,spliter=spliter,
                                   skip_connection=True,  use_order_books=False,
                                   order_book_size=OBs, tboard="TEST", IDn=IDn)
        return model_wrapper
    
    @pytest.mark.parametrize(
        "country, dataset",
        #[["FR", "Lyon"], ["DE", "Munich"], ["NL", "Lahaye"], ["BE", "Bruges"]])
        [["FR", "Lyon"]])
    @pytest.mark.parametrize("OBs", [20])
    @pytest.mark.parametrize(
        "IDn", [8, 9, 10, 11, 12, 13])
    def test_routine(self, country, dataset, OBs, IDn):
        model_wrapper = self.get_model(country, dataset, OBs, IDn)
        X, Y = model_wrapper.load_train_dataset()
        X = X[:N_SAMPLES, :]
        Y = Y[:N_SAMPLES, :]
        
        # Check that the input data shapes is well computed
        a = model_wrapper.N_X
        b = X.shape[1]
        assert a == b, f"model_wrapper.N_X != X.shape[1], {a}!={b}"
        a = model_wrapper.N_DATA
        b = X[:, model_wrapper.x_indices].shape[1]
        assert a == b, f"model_wrapper.N_DATA != X[:, model_wrapper.x_indices].shape[1], {a}!={b}"        
        a = model_wrapper.N_INPUT
        b = X.shape[1]
        if (not model_wrapper.use_order_books) and model_wrapper.OB_in_X:
            b -= 72 * model_wrapper.OBs
        assert a == b, f"model_wrapper.N_INPUT != N_INPUT - 72OBs * predict order books, {a}!={b}"        
        
        # Check that the output data shapes is well computed        
        a = model_wrapper.N_Y
        b = Y.shape[1]
        assert a == b, f"model_wrapper.N_Y != Y.shape[1], {a}!={b}"
        a = model_wrapper.N_PRICES
        b = Y[:, model_wrapper.y_indices].shape[1]
        assert a == b, f"model_wrapper.N_PRICES != Y[:, model_wrapper.y_indices].shape[1], {a}!={b}"
        a = model_wrapper.N_OUTPUT
        b = Y.shape[1]
        if model_wrapper.predict_order_books:
            b -= model_wrapper.N_PRICES
            
        assert a == b, f"model_wrapper.N_OUTPUT != Y.shape[1] - N_PRICES if predict ob, {a}!={b}"
        
        # Check that the x an y indices are coherent.
        a = len(model_wrapper.x_indices)
        b = model_wrapper.N_DATA
        assert a == b, f"len(model_wrapper.x_indices) != model_wrapper.N_DATA, {a}!={b}"
        a = len(model_wrapper.y_indices)
        b = model_wrapper.N_PRICES
        assert a == b, f"len(model_wrapper.y_indices) != model_wrapper.N_PRICES, {a}!={b}"
        
        # Check the input OB indices coherence
        a = sum([len(model_wrapper.v_indices),
                 len(model_wrapper.po_indices),
                 len(model_wrapper.p_indices)])
        b = model_wrapper.N_X - model_wrapper.N_DATA
        assert a == b, f"Sum of OB indices != N_X - N_DATA, {a}!={b}"

        # Check the output OB indices coherence
        a = sum([len(model_wrapper.yv_indices),
                 len(model_wrapper.ypo_indices),
                 len(model_wrapper.yp_indices)])
        b = model_wrapper.N_Y - model_wrapper.N_PRICES
        assert a == b, f"Sum of Y OB indices != NY - NPRICES, {a}!={b}"

        ######### TEST MAKE
        ptemp = model_wrapper.params()
        ptemp["early_stopping"] = None
        ptemp["n_epochs"] = N_EPOCHS
        ptemp["OB_plot"] = os.path.join(model_wrapper.logs_path)
        ptemp["profile"] = True
        ptemp["batch_size"] = BATCH

        try:
            regr = model_wrapper.make(ptemp)
        except Exception as e:
            pytest.fail(f"regr making has failed {e}")

        try:
            regr.fit(X, Y)
        except Exception as e:
            pytest.fail(f"regr fitting has failed {e}")            

        (Xtr, Ytr), (Xv, Yv) = model_wrapper.spliter(X, Y)
        yhat = model_wrapper.predict_val(regr, Xv)

        assert yhat.shape[1] == model_wrapper.N_OUTPUT

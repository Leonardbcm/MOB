from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from torch.profiler import profile, record_function, ProfilerActivity
import torch.autograd.profiler as profiler
import matplotlib.pyplot as plt, time

from src.models.spliter import MySpliter
import src.models.model_utils as mu
from src.models.scalers import SignOBNScaler
from src.models.model_wrapper import *
from src.models.torch_models.obn import OrderBookNetwork
from src.models.torch_models.weight_initializers import *
from src.models.torch_models.torch_obn import filter_key_averages

from src.models.samplers.combined_sampler import combined_sampler, list_combined_sampler
from src.models.samplers.structure_sampler import structure_sampler
from src.models.samplers.obn_structure_sampler import obn_structure_sampler
from src.models.samplers.regularization_sampler import regularization_sampler
from src.models.samplers.discrete_log_uniform import discrete_loguniform
from src.models.samplers.weight_initializer_samplers import wi_sampler, bi_sampler, wibi_sampler


class TorchWrapper(ModelWrapper):
    """
    Base wrapper for all torch-based models
    """
    def __init__(self, prefix, dataset_name, spliter=None, country="",
                 skip_connection=False, use_order_books=False,
                 order_book_size=20, IDn=0, alpha=1, beta=0, gamma=0, tboard=""):
        ModelWrapper.__init__(
            self, prefix, dataset_name, spliter=spliter, country=country,
            skip_connection=skip_connection, use_order_books=use_order_books,
            order_book_size=order_book_size, alpha=alpha, beta=beta, gamma=gamma,
            IDn=IDn)
        self.tboard = tboard

    def validation_prediction_path(self):
        return os.path.join(mu.folder(self.dataset_name),
                            self.ID + "_validation_predictions.csv")
    
    def test_prediction_path(self):
        return os.path.join(mu.folder(self.dataset_name),
                            self.ID + "_test_predictions.csv")
    
    @property
    def logs_path(self):
        return os.path.join(
            os.environ["MOB"], "logs", self.tboard, self.ID)

    def checkpoint_path(self, version):        
        return os.path.join(self.logs_path, version, "checkpoints")

    def checkpoint_file(self, path, epoch=None, step=None):
        if not(epoch is None and step is None):
            filename = "epoch={epoch}-step={step}.ckpt"
        else:
            checkpoints = os.listdir(path)
            
            epochs = []
            for checkpoint in checkpoints:
                epochs += [self.stopped_epoch_from_checkpoint(checkpoint)]
                
            most_recent = np.argmax(np.array(epochs))
            filename = checkpoints[most_recent]            
                
        return filename

    @property
    def latest_version(self):
        """
        Get the most recent version = version with the highest number
        """
        path = self.logs_path
        all_versions = os.listdir(path)
        all_v = np.array([int(v.split("_")[1]) for v in all_versions])
        return all_versions[np.argmax(all_v)]
        

    @property
    def highest_version(self):
        """
        Get the highest version = version with the biggest number of epochs
        """
        path = self.logs_path
        epochs = []
        versions = os.listdir(path)
        for version in versions:
            try:
                cur_epoch = self.get_stopped_epoch(version)
            except:
                cur_epoch = 0
            epochs += [cur_epoch]
        ind = np.argmax(np.array(epochs))
        return versions[ind]

    def get_stopped_epoch(self, version):
        filename = self.checkpoint_file(self.checkpoint_path(version))
        return self.stopped_epoch_from_checkpoint(filename)
        
    def stopped_epoch_from_checkpoint(self, filename):
        return int(filename.split('epoch=')[1].split('-step')[0])        

    def latest_checkpoint(self, version):
        path = self.checkpoint_path(version)
        filename = self.checkpoint_file(path)
        return os.path.join(path, filename)

    def load_refit(self, regr, X, Y, version):
        """
        Reload the model's state from a checkpoint and use it for training
        """
        train_loader, val_loader = regr.steps[1][1].create_trainer(X, Y)
        
        trainer = regr.steps[1][1].trainer
        model = regr.steps[1][1].model        
        trainer.fit(model, ckpt_path=self.latest_checkpoint(version),
                    train_dataloaders=train_loader, val_dataloaders=val_loader)

    def load_predict(self, regr, X, version):
        """
        Reload the model's state from a checkpoint and use it for prediction
        """
        test_loader = regr.steps[1][1].create_trainer(X)
        
        trainer = regr.steps[1][1].trainer
        model = regr.steps[1][1].model        
        yhat = trainer.predict(
            model, test_loader,ckpt_path=self.latest_checkpoint(version))           
        
    def get_real_prices(self, dataset="validation"):
        """
        Reload the labels and align them with the corresponding dates
        """
        if dataset == "validation":
            X, Y = self.load_train_dataset(OB=False)
            (_, _), (Xv, Y) = self.spliter(X, Y)
            _, dates = self.spliter(self.train_dates)
        else:
            _, Y = self.load_test_dataset(OB=False)
            dates =  self.test_dates
        
        if self.predict_order_books:
            Y = Y[:, -len(self.label):]    
    
        return pandas.DataFrame(Y, index=dates)

    def get_predictions(self, dataset="validation"):
        if dataset == "validation":
            path = self.validation_prediction_path()
        else:
            path = self.test_prediction_path()
            
        try:
            ypred = pandas.read_csv(path, index_col="Unnamed: 0")
        except:
            return None
        ypred.index = [datetime.datetime.strptime(d, "%Y-%m-%d")
                       for d in ypred.index]
        ypred.index = [datetime.date(d.year, d.month, d.day) for d in ypred.index]
        return ypred

    def plot_forecasts(self, dataset="validation"):
        # Plot forecasts
        ypred = self.get_predictions(dataset=dataset)
        ytrue = self.get_real_prices(dataset=dataset)
        
        plt.plot(ypred.values.reshape(-1), label="Predictions")
        plt.plot(ytrue.values.reshape(-1), label="Real Prices")
        plt.legend()
        plt.grid("on")
        plt.title(f"{self.ID} {dataset} results")
        plt.show()        

    def compute_metrics(self):
        # Reload predictions
        validation_predictions = self.get_predictions()
        test_predictions = self.get_predictions(dataset="test")

        if validation_predictions is None or test_predictions is None:
            return {}
        
        # Reload real prices
        validation_prices = self.get_real_prices()
        test_prices = self.get_real_prices(dataset="test")      
        line = {
            "country" : self.country,
            "order_book_size" : self.order_book_size,
            "alpha" : self.alpha,
            "beta" : self.beta,
            "gamma" : self.gamma,            
            "val_mae" : self.mae(
                validation_prices.values, validation_predictions.values),
            "val_dae" : self.dae(
                validation_prices.values, validation_predictions.values),
            "val_smape" : self.smape(
                validation_prices.values, validation_predictions.values),
            "val_ACC" : self.ACC(
                validation_prices.values, validation_predictions.values),
            "test_mae" : self.mae(test_prices.values, test_predictions.values),
            "test_dae" : self.dae(test_prices.values, test_predictions.values),
            "test_smape" : self.smape(test_prices.values, test_predictions.values),
            "test_ACC" : self.ACC(test_prices.values, test_predictions.values)    
        }
        return line

    def load_coeffs(self):
        coeffs = np.load(os.path.join(os.environ["MOB"], "coeffs.npy"))

        countries = np.array(["FR", "DE", "BE", "NL"])
        datasets = np.array(["Lyon", "Munich", "Bruges", "Lahaye"])
        order_book_sizes = np.array([20, 50, 100, 250])

        x = np.where(countries == self.country)[0][0]
        y = np.where(order_book_sizes == self.order_book_size)[0][0]

        self.coefs = coeffs[x, y]

        
class OBNWrapper(TorchWrapper):
    """
    Wrapper for all predict order books then optimize models
    """    
    def __init__(self, prefix, dataset_name, pmin=-500, pmax=3000,
                 spliter=None, country="", skip_connection=False,                 
                 use_order_books=False, order_book_size=20, IDn=0,
                 alpha=1, beta=0, gamma=0, tboard=""):
        TorchWrapper.__init__(
            self, prefix, dataset_name, spliter=spliter, country=country,
            skip_connection=skip_connection, use_order_books=use_order_books,
            IDn=IDn, order_book_size=order_book_size,
            alpha=alpha, beta=beta, gamma=gamma, tboard=tboard)
        if spliter is None: spliter = MySpliter(0.25)
        
        self.pmin = pmin
        self.pmax = pmax        
        self.spliter = spliter
        self.external_spliter = None        
        self.validation_mode = "internal"
        self.prev_label = []

        ############# Compute SHAPES            
        self.N_DATA = len(self.columns)
        self.N_X = self.N_DATA
        if self.OB_in_X:
            self.N_X += self.OBs * 72

        self.N_INPUT = self.N_DATA
        if self.OB_in_input:
            self.N_INPUT += self.OBs * 72        
        
        self.N_PRICES = len(self.label)

        self.N_OUTPUT = 0
        if self.OB_in_output:
            self.N_OUTPUT += self.OBs * 72
        if self.p_in_output:
            self.N_OUTPUT += self.N_PRICES
        
        self.N_Y = 0
        if self.OB_in_Y:
            self.N_Y += self.OBs * 72
        if self.p_in_Y:
            self.N_Y += self.N_PRICES
            
        ############ Init indices
        self.init_indices()

    @property
    def ID(self):
        ID = self.country + "_"

        if (self.alpha == 1) and (self.gamma == 0) and (self.beta == 0):
            model_number = 1
        if (self.alpha == 0) and (self.gamma == 1) and (self.beta == 0):
            model_number = 2
        if (self.alpha == 0.5) and (self.gamma == 0.5) and (self.beta == 0):
            model_number = 3            
        if (self.alpha == 0) and (self.gamma == 0) and (self.beta == 1):
            model_number = 4
        if (self.alpha == 0) and (self.gamma == 0.5) and (self.beta == 0.5):
            model_number = 5
        if (self.alpha == 0.5) and (self.gamma == 0) and (self.beta == 0.5):
            model_number = 6
        if (self.alpha == 1/3) and (self.gamma == 1/3) and (self.beta == 1/3):
            model_number = 7
            
        ID += str(model_number) + "_" + str(self.order_book_size)
        return ID
        
    def params(self):
        self.load_coeffs()
        
        # Use the specified order book size (for loading and predicting)
        OBs = 100
        if self.use_order_books or self.skip_connection or self.predict_order_books:
            OBs = self.order_book_size

        # Update the search bounds using the scalers
            
        default_params = {                       
            # Model Architecture
            "skip_connection" : self.skip_connection,
            "use_order_books" : self.use_order_books,
            "order_book_size" : self.order_book_size,
            "predict_order_books" : self.predict_order_books,
            
            # Loss Coefficients
            "alpha" : self.alpha,
            "beta" : self.beta,
            "gamma" : self.gamma,
            "coefs" : self.coefs,            

            # Network architecture
            "N_OUTPUT" : self.N_OUTPUT,
            "N_Y" : self.N_Y,                        
            "N_PRICES" : self.N_PRICES,
            "N_INPUT" : self.N_INPUT,
            "N_X" : self.N_X,
            "N_DATA" : self.N_DATA,             
            "NN1" : (888, ),
            "OBN" : (37, ),
            "OBs" : OBs,
            "k" : 100,
            "niter" : 20, 
            "batch_norm" : True,
            "dropout" : 0.1,            

            # Solver Parameters
            "batch_solve" : True,            
            "pmin" : self.pmin,
            "pmax" : self.pmax,
            "step" : 0.01,
            "mV" : 0.1,
            "check_data" : False,

            # Log params
            "store_OBhat" : False,
            "store_val_OBhat" : False,
            "store_losses" : False,
            "OB_plot" : False,
            "tensorboard" : self.tboard,
            "ID" : self.ID,            
            "profile" : False,
            "log_every_n_steps" : 1,
            
            # Scaling Parameters
            "scaler" : "Standard",
            "transformer" : "Standard",
            "weight_initializers" : [BiasInitializer(
                "normal", 30, 40, self.pmin, self.pmax)],
            "scale" : "",
            
            # Training Params            
            "spliter" : self.spliter,
            "n_epochs" : 1000,
            "batch_size" : 30,
            "early_stopping" : "lightning",
            "early_stopping_alpha" : 20,
            "early_stopping_patience" : 100,
            "very_early_stopping" : False,
            "shuffle_train" : True,

            # Optimizer Params
            "n_cpus" : -1,

            # Indices
            "y_indices" : self.y_indices,
            "yv_indices" : self.yv_indices,
            "ypo_indices" : self.ypo_indices,
            "yp_indices" : self.yp_indices,
        
            "x_indices" : self.x_indices,
            "v_indices" : self.v_indices,
            "po_indices" : self.po_indices,
            "p_indices" : self.p_indices,
        }       
        
        return default_params

    def make(self, ptemp):
        base_scaler = ptemp["scaler"]
        base_transformer = ptemp["transformer"]
        
        scaler = SignOBNScaler(
            base_scaler, ptemp["OBs"], self.N_X, self.N_DATA,
            self.x_indices, self.v_indices, self.po_indices, self.p_indices,
            spliter=self.spliter, nh=24)
        
        transformer =  SignOBNScaler(
            base_transformer, ptemp["OBs"], self.N_Y, self.N_PRICES,
            self.y_indices, self.yv_indices, self.ypo_indices, self.yp_indices,
            spliter=self.spliter, nh=24)
        
        ptemp_ = copy.deepcopy(ptemp)
        ptemp_["transformer_"] = transformer 

        # Disable WI if skip connection
        if self.skip_connection:
            print("Disabling weight_initializers because skip_connection=True")
            ptemp_["weight_initializers"] = []

        # Disabling OB logging if no OB in the network!
        if self.gamma == 0:
            ptemp_["store_OBhat"] = False
            ptemp_["store_val_OBhat"] = False
            ptemp_["OB_plot"] = False
            print("Disabling OB logging because no OB will be produced")        

        # Set profiler if specified
        if ptemp_["profile"]:
            self.set_profiler()
            ptemp_["profiler"] = self.profiler
        else:
            ptemp_["profiler"] = False
            
        # OBN need to access the scalers for scaling OBN
        model = OrderBookNetwork("test", ptemp_)
        pipe = make_pipeline(scaler, model)
        return pipe

    def set_profiler(self):
        def trace_handler(p):
            out = p.key_averages(group_by_stack_n=5)
            df = filter_key_averages(out)
            df.to_csv(os.path.join(
                self.logs_path, self.latest_version, f"table_{self.ID}.csv"))
            torch.profiler.tensorboard_trace_handler(
                os.path.join(self.logs_path, self.latest_version))(p)
    
        self.profiler = profile(
            activities=[ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(
                wait=1,warmup=1,active=1,repeat=1),
            on_trace_ready=trace_handler)

    def get_search_space(self, n=None, fast=False, stop_after=-1):
        space = {
            "structure" : combined_sampler(
                [
                    obn_structure_sampler(n, 1, 0,  25, 60),
                    obn_structure_sampler(n, 1, 1,  25, 60),
                    obn_structure_sampler(n, 1, 2,  25, 60),

                    obn_structure_sampler(n, 2, 0,  25, 60),
                    obn_structure_sampler(n, 2, 1,  25, 60),
                    obn_structure_sampler(n, 2, 2,  25, 60),                    
                ],
                weights = [2, 2, 2, 1, 1, 1]),
            "OBs" : discrete_loguniform(50, 500),
            "weight_initializer" : list_combined_sampler(
                [
                    wibi_sampler(self.pmin, self.pmax),
                    wi_sampler(),
                    bi_sampler(self.pmin, self.pmax),
                    []
                ],
                weights = [4, 1, 2, 3]
            ),
            "batch_norm" : stats.bernoulli(0.5),
            "batch_size" : discrete_loguniform(10, n+1),            
            "scaler" : ["BCM", "Standard", "Median", "SinMedian"],
            "transformer" : combined_sampler(
                ["BCM", "Standard", "Median", "SinMedian", ""],
                weights = [1, 1, 1, 1, 4],
            ),
            "stop_after" : [stop_after],
            "log_every_n_steps" : [1],
            "tensorboard" : ["OBN_GRIDSEARCH"],
            "very_early_stopping" : [40],
        }
        if fast:
            space["n_epochs"] = [2]
            space["early_stopping"] = [""]               
        return space

    def map_dict(self):
        orig = TorchWrapper.map_dict(self)
        orig.update({"structure" :
                     {
                         "OBN" : (mu.neurons_per_layer_to_string,
                                  mu.neurons_per_layer_from_string),
                         "NN1" : (mu.neurons_per_layer_to_string,
                                  mu.neurons_per_layer_from_string),
                     },
                     "weight_initializer" : (mu.weight_initializer_to_string,
                                             mu.weight_initializer_from_string) 
        })
        return orig

    def _load_dataset(self, path, dataset, OB=True):
        ############# Load regular data
        df = pandas.read_csv(path)
        datetimes = [datetime.datetime.strptime(
            d, "%Y-%m-%d") for d in df.period_start_date]
        df.index = datetimes

        dates = [datetime.date(d.year, d.month, d.day) for d in datetimes]
        df.drop(columns="period_start_date", inplace=True)
        
        if dataset == "train":
            self.train_dates = np.array(dates)
        if dataset == "test":
            self.test_dates = np.array(dates)      

        labels = self.label.copy()
        columns = self.columns.copy()
        ############# load OB if needed
        # OB are needed if they are part of the labels OR
        # we decided to use them and the are needed for solving
        if (self.gamma > 0) or (self.beta > 0  and (self.skip_connection or self.use_order_books)) and OB:            
            OB_features, OB_columns, OB_lab, OB_labels= self.load_order_books()

            # Add them to the features if we specified to use them
            if (self.skip_connection or self.use_order_books):
                df = df.join(OB_features)
                columns += OB_columns            

            # Remove the first day because there are no previous order books
            if (dataset == "train"):            
                df.drop(datetime.date(2016, 1, 1), inplace=True)

            # Add them to the labels only if they are part of the labels.
            if self.gamma > 0:
                df = df.join(OB_lab)
                labels = OB_labels + labels

        Y = df.loc[:, labels].values
        X = df.loc[:, columns].values        
        return X, Y

    def load_order_books(self):
        order_book_path = os.path.join(
            os.environ["MOB"],"curves",
            f"{self.country}_{self.order_book_size}.csv")
        OB = pandas.read_csv(order_book_path)

        date_col = "period_start_date" 
        OB.index = [datetime.datetime.strptime(
            d, "%Y-%m-%d") for d in OB.loc[:, date_col]]
        OB.drop(columns=date_col, inplace=True)
        
        variables = ["V", "Po", "P"]
        OBs = int(self.order_book_size)
        OB_columns = [f"OB_{h}_{v}_{ob}_past_1" for h in range(24)
                      for v in variables for ob in range(OBs)]
        OB_features = OB.loc[:, OB_columns]
        
        OB_labels = [f"OB_{h}_{v}_{ob}" for h in range(24)
                     for v in variables for ob in range(OBs)]
        OB_lab = OB.loc[:, OB_labels]

        return OB_features, OB_columns, OB_lab, OB_labels
    
    def load_train_dataset(self, OB=True):
        return self._load_dataset(self.train_dataset_path(), "train", OB=OB)
    
    def load_test_dataset(self, OB=True):
        return self._load_dataset(self.test_dataset_path(), "test", OB=OB)

    def past_price_col_indices(self):
        return np.array([np.where(np.array(self.columns) == f"{self.country}_price_{h}_past_1")[0][0] for h in range(24)])
    
    def load_and_reshape(self):
        self.init_indices()
        
        # Load Data
        X, Y = self.load_train_dataset()
        OB_features = X[:, -24*self.order_book_size*3:]
        OB_labels = Y[:, :24*self.order_book_size*3]
        past_prices = X[:, self.past_price_col_indices()]
        real_prices = Y[:, -24:]
        
        # Reshape everything
        Yv = OB_labels[:, self.yv_indices].reshape(
            -1, self.order_book_size)
        Yhatv = OB_features[:, self.yv_indices].reshape(
            -1, self.order_book_size)
        
        Ypo = OB_labels[:, self.ypo_indices].reshape(
            -1, self.order_book_size)
        Yhatpo = OB_features[:, self.ypo_indices].reshape(
            -1, self.order_book_size)
        
        Yp = OB_labels[:, self.yp_indices].reshape(
            -1, self.order_book_size)
        Yhatp = OB_features[:, self.yp_indices].reshape(
            -1, self.order_book_size)

        past_prices = past_prices.reshape(-1)
        real_prices = real_prices.reshape(-1)

        df = pandas.read_csv(self.train_dataset_path())
        datetimes = [datetime.datetime.strptime(
            d, "%Y-%m-%d") for d in df.period_start_date]
        df.index = datetimes

        datetimes = [d+datetime.timedelta(hours=h) for d in datetimes
                     for h in range(24)]
        return Yv, Yhatv, Ypo, Yhatpo, Yp, Yhatp, past_prices, real_prices,datetimes

    def default_indices(self):
        """
        Computes the default indices if no OB are used
        """
        self.y_indices = np.arange(self.N_PRICES)
        self.yv_indices = np.array([], dtype=bool) 
        self.ypo_indices = np.array([], dtype=bool)
        self.yp_indices = np.array([], dtype=bool)        

        self.x_indices = np.arange(self.N_DATA)
        self.v_indices = np.array([], dtype=bool) 
        self.po_indices = np.array([], dtype=bool)
        self.p_indices = np.array([], dtype=bool)

    def init_indices(self):
        """
        Computes the indices of the OB in the input data but also in the labels.
        Also computes the indices of each OB component.
        """
        self.default_indices()        
        # OB indices in the input data
        if (self.skip_connection or self.use_order_books) and ((self.beta > 0) or (self.gamma > 0)):
            start = self.N_DATA
            self.v_indices,self.po_indices,self.p_indices=self.compute_indices(
                self.N_DATA)
            self.OB_indices = np.concatenate((
                self.v_indices,self.po_indices,self.p_indices))            
            
        # OB indices in the labels
        if self.gamma > 0:
            self.yv_indices,self.ypo_indices,self.yp_indices=self.compute_indices(0)
            self.y_indices = np.arange(self.N_Y - self.N_PRICES, self.N_Y)
            self.yOB_indices = np.concatenate((
                self.yv_indices,self.ypo_indices,self.yp_indices))

    def compute_indices(self, start):
        """
        Given a starting position, compute the indices of individual parts of the OB
        """
        v_indices = np.array([start + 3*self.OBs*h+i for h in range(24)
                              for i in range(self.OBs)])
        po_indices = np.array([v + self.OBs for v in v_indices])
        p_indices = np.array([po + self.OBs for po in po_indices])
        return v_indices, po_indices, p_indices
            
    def refit(self, regr, X, y, epochs=1):
        model = regr.steps[1][1]
        scaler = regr.steps[0][1]        
        model.refit(scaler.transform(X), y, epochs=epochs)

    def string(self):
        return "OBN"

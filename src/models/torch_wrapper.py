from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor

from src.models.spliter import MySpliter
import src.models.model_utils as mu
from src.models.model_wrapper import *
from src.models.torch_models.obn import OrderBookNetwork
from src.models.torch_models.weight_initializers import *

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
                 order_book_size=20, separate_optim=False, tboard=""):
        ModelWrapper.__init__(
            self, prefix, dataset_name, spliter=spliter, country=country,
            skip_connection=skip_connection, use_order_books=use_order_books,
            order_book_size=order_book_size, separate_optim=separate_optim)
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
            filename = checkpoints[-1]
        return filename

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
                    
        

class OBNWrapper(TorchWrapper):
    """
    Wrapper for all predict order books then optimize models
    """    
    def __init__(self, prefix, dataset_name, pmin=-500, pmax=3000,
                 spliter=None, country="", skip_connection=False,                 
                 use_order_books=False, order_book_size=20,
                 separate_optim=False, tboard=""):
        TorchWrapper.__init__(
            self, prefix, dataset_name, spliter=spliter, country=country,
            skip_connection=skip_connection, use_order_books=use_order_books,
            order_book_size=order_book_size, separate_optim=separate_optim,
            tboard=tboard)
        if spliter is None: spliter = MySpliter(0.25)
        
        self.pmin = pmin
        self.pmax = pmax        
        self.spliter = spliter
        self.external_spliter = None        
        self.validation_mode = "internal"
        self.prev_label = []

    @property
    def ID(self):
        ID = self.country + "_"
        
        if self.separate_optim:
            ID += "SO=TRUE"
        else:
            ID += "SO=FALSE"

        ID += "_"
        
        if self.skip_connection:
            ID += "SC=TRUE"
        else:
            ID += "SC=FALSE"

        ID += "_"            
        if self.use_order_books:
            ID += "UOB=TRUE"
        else:
            ID += "UOB=FALSE"

        ID += "_" + str(self.order_book_size)
        return ID           
        
    def params(self):
        n_output = len(self.label)        
        if self.separate_optim:
            n_output = self.order_book_size * 24 * 3        

        # Use the specified order book size (for loading and predicting)
        OBs = 100
        if self.use_order_books or self.skip_connection or self.separate_optim:
            OBs = self.order_book_size
            
        default_params = {                       
            # Model Architecture
            "skip_connection" : self.skip_connection,
            "use_order_books" : self.use_order_books,
            "order_book_size" : self.order_book_size,
            "separate_optim" : self.separate_optim,

            # Network architecture
            "N_OUTPUT" : n_output,            
            "N_PRICES" : len(self.label),
            "NN1" : (888, ),
            "OBN" : (37, ),
            "OBs" : OBs,
            "k" : 100,
            "niter" : 30, 
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
            "tensorboard" : self.tboard,
            "ID" : self.ID,            
            "profile" : False,
            "log_every_n_steps" : 1,
            
            # Scaling Parameters
            "scaler" : "Standard",
            "transformer" : "Standard",
            "OB_transformer" : "Standard",            
            "weight_initializers" : [BiasInitializer(
                "normal", 30, 40, self.pmin, self.pmax)],
            "scale" : "Clip-Sign",

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
            "criterion" : "HuberLoss",
            "n_cpus" : -1,
        }
                    
        # Disable WI if skip connection
        if self.skip_connection:
            default_params["weight_initializers"] = []
        if self.separate_optim:
            default_params["criterion"] = "smape"
        return default_params

    def make(self, ptemp):
        scaler, transformer, OB_transformer, ptemp_ = self.prepare_for_make(ptemp)
        ptemp_["transformer"] = transformer
        ptemp_["OB_transformer"] = OB_transformer        
        
        # OBN need to access the scalers for scaling OBN
        model = OrderBookNetwork("test", ptemp_)
        pipe = make_pipeline(scaler, model)
        return pipe

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

    def _load_dataset(self, path, dataset):
        # Load regular data        
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
        
        # load OB if needed
        if self.skip_connection or self.use_order_books or self.separate_optim:
            order_book_path = os.path.join(
                os.environ["MOB"],"curves",
                f"{self.country}_{self.order_book_size}.csv")
            OB = pandas.read_csv(order_book_path)            
            OB.index = [datetime.datetime.strptime(
                d, "%Y-%m-%d") for d in OB.period_start_date]
            OB.drop(columns="period_start_date", inplace=True)
                        
            variables = ["V", "Po", "P"]
            OBs = int(self.order_book_size)
            OB_columns = [f"OB_{h}_{v}_{ob}_past_1" for h in range(24)
                          for v in variables for ob in range(OBs)]
            OB_features = OB.loc[:, OB_columns]
            
            df = df.join(OB_features)
            self.columns += OB_columns
            
            if (dataset == "train"):
                # Remove the first day because there are no previous order books
                df.drop(datetime.date(2016, 1, 1), inplace=True)
                
            Y = df.loc[:, self.label].values
            X = df.drop(columns=self.label).values    
            if self.separate_optim:
                OB_labels = [f"OB_{h}_{v}_{ob}" for h in range(24)
                             for v in variables for ob in range(OBs)]
                OB_lab = OB.loc[:, OB_labels]
                df = df.join(OB_lab)
                
                Y = df.loc[:, OB_labels + self.label].values
                X = df.drop(columns=OB_labels + self.label).values
        else:
            Y = df.loc[:, self.label].values
            X = df.drop(columns=self.label).values
        return X, Y

    def load_train_dataset(self):
        return self._load_dataset(self.train_dataset_path(), "train")
    
    def load_test_dataset(self):
        return self._load_dataset(self.test_dataset_path(), "test")

    def predict_order_books(self, regr, X):
        model = regr.steps[1][1]
        scaler = regr.steps[0][1]
        
        OBhat = model.predict_order_books(scaler.transform(X))
        return OBhat

    def refit(self, regr, X, y, epochs=1):
        model = regr.steps[1][1]
        scaler = regr.steps[0][1]
        
        model.refit(scaler.transform(X), y, epochs=epochs)

    def string(self):
        return "OBN"

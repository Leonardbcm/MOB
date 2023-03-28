import torch, torch.nn as nn, pytorch_lightning as pl, os, numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error

from src.models.torch_models.ob_datasets import EPFDataset
from src.models.torch_models.torch_obn import SolvingNetwork
from src.models.torch_models.callbacks import *

class OrderBookNetwork(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible model
    """
    def __init__(self, name, model_):
        self.name = name
        self.model_ = model_
        
        self.dtype = torch.float32        

        # Architecture parameters
        self.skip_connection = model_["skip_connection"]
        self.use_order_books = model_["use_order_books"]
        self.separate_optim = model_["separate_optim"]
        # Should be equal to OBs or 0
        self.order_book_size = model_["order_book_size"]
        
        # Network structure parameters
        self.NN1 = model_["NN1"]
        self.OBN = model_["OBN"]        
        self.OBs = model_["OBs"]
        self.N_OUTPUT = model_["N_OUTPUT"]
        self.N_PRICES = model_["N_PRICES"]        
        self.dropout = model_["dropout"]
        
        # Other network parameters
        self.transformer = model_["transformer"]
        self.OB_transformer = model_["OB_transformer"]        
        self.weight_initializers = model_["weight_initializers"]    
        self.scale = model_["scale"]        
        self.batch_norm = model_["batch_norm"]        

        # Solver parameters
        self.k = model_["k"]
        self.niter = model_["niter"]
        self.batch_solve = model_["batch_solve"]
        self.pmin = model_["pmin"]
        self.pmax = model_["pmax"]
        self.step = model_["step"]
        self.mV = model_["mV"]
        self.check_data = model_["check_data"]

        # Training parameters
        self.n_epochs = model_["n_epochs"]
        self.batch_size = model_["batch_size"]
        self.shuffle_train = model_["shuffle_train"]
        self.criterion = model_["criterion"]
        self.spliter = model_["spliter"]        

        # Callbacks parameters
        self.early_stopping = model_["early_stopping"]
        self.early_stopping_alpha = model_["early_stopping_alpha"]
        self.early_stopping_patience = model_["early_stopping_patience"]
        self.very_early_stopping = model_["very_early_stopping"]
        
        # Logging aprameters
        self.store_losses = model_["store_losses"]
        self.tensorboard = model_["tensorboard"]
        self.ID = model_["ID"]
        # For saving tboard logs and checkpoints
        self.logdir = os.path.join(
            os.environ["MOB"], "logs", self.tensorboard)    
        self.profile = model_["profile"]
        self.store_OBhat = model_["store_OBhat"]
        self.store_val_OBhat = model_["store_val_OBhat"]
        self.log_every_n_steps = model_["log_every_n_steps"]        

        # Parallelization params
        self.n_cpus = model_["n_cpus"]
        if self.n_cpus == -1:
            self.n_cpus_ = os.cpu_count()
        else:
            self.n_cpus_ = self.n_cpus
            
        torch.set_num_threads(self.n_cpus_)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():            
            setattr(self, parameter, value)
        return self

    def update_params(self, input_shape):
        """
        Reset the network networks:
         (otpimizer, training state, callbacks, etc)
        To call everytime an hyperparameter changes!
        """
        
        # Correct the input shape if order books where loaded but unused
        if not self.use_order_books and (self.skip_connection or self.separate_optim):
            input_shape = input_shape - self.OBs * 24 * 3
            
        self.model = self.create_network(input_shape=input_shape)

        # Set callbacks        
        self.callbacks = []
        if self.store_losses:
            self.callbacks += [StoreLosses()]
        if self.store_val_OBhat:
            self.callbacks += [ValOBhat(self.store_val_OBhat)]
        if self.store_OBhat:
            self.callbacks += [StoreOBhat(self.store_OBhat)]
        self.early_stopping_callbacks()

    def create_trainer(self, X, Y=None, verbose=0):
        # TRAIN CASE
        if Y is not None:
            # Prepare the data : scale and make datalaoders
            train_loader, val_loader = self.prepare_for_train(X, Y)
            res = train_loader, val_loader
        else:
            test_loader = self.prepare_for_test(X)
            res = test_loader

        # Instanciate the model
        self.update_params(input_shape=X.shape[1])
        print(self.model)
        
        # Create the trainer
        self.trainer = pl.Trainer(
            max_epochs=self.n_epochs, callbacks=self.callbacks,
            logger=TensorBoardLogger(self.logdir, name=self.ID),
            enable_checkpointing=True,
            log_every_n_steps=self.log_every_n_steps,
            default_root_dir=self.logdir, enable_progress_bar=True)
        
        return res
                
    ###### METHODS FOR SKLEARN
    def fit(self, X, Y, verbose=0):
        train_loader, val_loader = self.create_trainer(X, Y, verbose=0)

        # Train   
        self.trainer.fit(self.model, train_dataloaders=train_loader,
                         val_dataloaders=val_loader)
        
    def predict(self, X):
        test_loader = self.prepare_for_test(X)
        predictions = self.trainer.predict(self.model, test_loader)
        ypred = torch.zeros(X.shape[0], self.N_PRICES)
            
        idx = 0
        for i, data in enumerate(predictions):
            bs = data.shape[0]                
            ypred[idx:idx+bs] = data
            idx += bs

        ypred = ypred.detach().numpy()
        ypred = self.transformer.inverse_transform(ypred)
        return ypred

    def predict_order_books(self, X):
        test_loader = self.prepare_for_test(X)
        predictions = [self.trainer.model.forward(batch, predict_order_books=True)
                       for batch in iter(test_loader)]

        ypred = torch.zeros(X.shape[0], 24, self.OBs, 3)
        idx = 0
        for i, data in enumerate(predictions):
            bs = int(data.shape[0]/24)
            ypred[idx:idx+bs] = data.reshape(bs, 24, self.OBs, 3)
            idx += bs

        ypred = ypred.detach().numpy()
        return ypred

    def score(self, X, y):
        yhat = self.predict(X)
        return mean_absolute_error(y, yhat)

    ######### HELPERS
    def early_stopping_callbacks(self):
        if self.early_stopping == "sliding_average":
            self.callbacks.append(
                EarlyStoppingSlidingAverage(
                    monitor="val_loss",
                    alpha=self.early_stopping_alpha,
                    patience=self.early_stopping_patience))
        if self.early_stopping == "lightning":
            self.callbacks.append(
                EarlyStopping(
                    monitor="val_loss", patience=self.early_stopping_patience))
            
    def create_network(self, input_shape):
        return SolvingNetwork(input_shape, self.NN1, self.OBs, self.OBN,
                              self.batch_norm, self.criterion,
                              self.N_OUTPUT, self.k, self.batch_solve, self.niter,
                              self.pmin, self.pmax, self.step, self.mV,
                              self.check_data, self.transformer,self.OB_transformer,
                              self.scale, self.weight_initializers, self.profile,
                              self.skip_connection, self.use_order_books,
                              self.separate_optim, self.N_PRICES, self.dropout)

    def split_prices_OB(self, y):
        """
        Compute the OB indices based on the specified orderbooksize
        Separate prices and order books from the labels.
        """
        stop = self.N_PRICES
        OB = y[:, :-stop].copy()
        y = y[:, -stop:].copy()
        return y, OB
    
    ######################## DATA FORMATING
    def prepare_for_train(self, X, y, transformers_are_fit=False):
        ((X, y), (Xv, yv)) = self.spliter(X, y)

        # Separate prices and order books labels.
        if self.separate_optim:
            prices, OB = self.split_prices_OB(y)
            prices_v, OB_v = self.split_prices_OB(yv)
            
            # We don't need transformed validation OB!
            if not transformers_are_fit:
                self.OB_transformer.fit(OB)
                
            y = self.OB_transformer.transform(OB)
            
            # Train prices are only used for fitting the transformer!!!
            if not transformers_are_fit:
                self.transformer.fit(prices)
                
            yv = self.transformer.transform(prices_v)        
        else:
            if not transformers_are_fit:
                self.transformer.fit(y)
                
            y = self.transformer.transform(y)                
            yv = self.transformer.transform(yv)
        
        NUM_WORKERS = self.n_cpus_
        train_dataset = EPFDataset(X, y, dtype=self.dtype, N_OUTPUT=self.N_OUTPUT)
        val_dataset = EPFDataset(Xv, yv, dtype=self.dtype, N_OUTPUT=self.N_PRICES)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size, shuffle=self.shuffle_train,
            num_workers=NUM_WORKERS)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size, shuffle=False,
            num_workers=NUM_WORKERS)

        return train_loader, val_loader 

    def prepare_for_test(self, X):        
        NUM_WORKERS = self.n_cpus_
        test_dataset = EPFDataset(X, dtype=self.dtype, N_OUTPUT=self.N_PRICES)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                 num_workers=NUM_WORKERS)
        return test_loader

    

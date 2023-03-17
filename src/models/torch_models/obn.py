import torch, torch.nn as nn, pytorch_lightning as pl, os, numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error

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

        # Copy params
        self.NN1 = model_["NN1"]
        self.OBN = model_["OBN"]        
        self.OBs = model_["OBs"]
        self.k = model_["k"]
        self.niter = model_["niter"]
        self.batch_solve = model_["batch_solve"]
        self.batch_norm = model_["batch_norm"]        
        self.pmin = model_["pmin"]
        self.pmax = model_["pmax"]
        self.step = model_["step"]
        self.mV = model_["mV"]
        self.check_data = model_["check_data"]
        
        self.n_epochs = model_["n_epochs"]
        self.batch_size = model_["batch_size"]
        self.shuffle_train = model_["shuffle_train"]
        self.early_stopping = model_["early_stopping"]
        self.early_stopping_alpha = model_["early_stopping_alpha"]
        self.early_stopping_patience = model_["early_stopping_patience"]

        self.criterion = model_["criterion"]
        self.N_OUTPUT = model_["N_OUTPUT"]
        self.spliter = model_["spliter"]
        self.store_OBhat = model_["store_OBhat"]
        
        self.store_losses = model_["store_losses"]
        self.tensorboard = model_["tensorboard"]
        self.logdir = os.path.join(os.environ["VOLTAIRE"], "logs")        
        
        # Used to transform the upper and lower bound!
        self.transformer = model_["transformer"]
        self.OB_weight_initializers = model_["OB_weight_initializers"]    
        self.scale = model_["scale"]

        # Parallelization params
        self.n_cpus = model_["n_cpus"]
        if self.n_cpus == -1:
            self.n_cpus_ = os.cpu_count()
        else:
            self.n_cpus_ = self.n_cpus

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
        # Instantiate the model
        self.model = self.create_network(input_shape=input_shape)

        # Set callbacks        
        self.callbacks = []
        if self.store_losses:
            self.callbacks += [StoreLosses()]
        if self.store_OBhat:
            #self.callbacks += [StoreOBhat(self.store_OBhat)]
            self.callbacks += [ValOBhat(self.store_OBhat)]
        self.early_stopping_callbacks()
    
    ###### METHODS FOR SKLEARN AND VOLTAIRE
    def fit(self, X, y, verbose=0):        
        # Prepare the data : scale and make datalaoders
        train_loader, val_loader = self.prepare_for_train(X, y)

        # Instanciate the model
        self.update_params(input_shape=X.shape[1])
        print(self.model)
        
        # Create the trainer
        self.trainer = pl.Trainer(
            max_epochs=self.n_epochs, callbacks=self.callbacks,
            logger=TensorBoardLogger(self.logdir, name=self.tensorboard),
            enable_checkpointing=False, log_every_n_steps=1,
            default_root_dir=self.logdir, enable_progress_bar=True)

        # Train   
        self.trainer.fit(self.model, train_dataloaders=train_loader,
                         val_dataloaders=val_loader)

    def refit(self, X, y, epochs=1):
        train_loader, val_loader = self.prepare_for_retrain(X, y)
        self.trainer = pl.Trainer(
            max_epochs=epochs, callbacks=self.callbacks,
            logger=TensorBoardLogger(self.logdir, name=self.tensorboard),
            enable_checkpointing=False, log_every_n_steps=1,
            default_root_dir=self.logdir, enable_progress_bar=True)
        
        self.trainer.fit(self.model, train_dataloaders=train_loader,
                         val_dataloaders=val_loader)
        
    def predict(self, X):
        test_loader = self.prepare_for_test(X)
        predictions = self.trainer.predict(self.model, test_loader)
        ypred = torch.zeros(X.shape[0], self.N_OUTPUT)
            
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

    def create_network(self, input_shape):
        return SolvingNetwork(input_shape, self.NN1, self.OBs, self.OBN,
                              self.batch_norm, self.criterion,
                              self.N_OUTPUT, self.k, self.batch_solve, self.niter,
                              self.pmin, self.pmax, self.step, self.mV,
                              self.check_data, self.transformer,
                              self.scale, self.OB_weight_initializers)
    
    ######################## DATA FORMATING
    def prepare_for_train(self, X, y):
        ((X, y), (Xv, yv)) = self.spliter(X, y)
        y = self.transformer.fit_transform(y)
        yv = self.transformer.transform(yv)
        
        NUM_WORKERS = self.n_cpus_
        train_dataset = EPFDataset(X, y, dtype=self.dtype, N_OUTPUT=self.N_OUTPUT)
        val_dataset = EPFDataset(Xv, yv, dtype=self.dtype, N_OUTPUT=self.N_OUTPUT)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size, shuffle=self.shuffle_train,
            num_workers=NUM_WORKERS)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size, shuffle=False,
            num_workers=NUM_WORKERS)

        return train_loader, val_loader

    def prepare_for_retrain(self, X, y):
        ((X, y), (Xv, yv)) = self.spliter(X, y)
        y = self.transformer.transform(y)
        yv = self.transformer.transform(yv)
        
        NUM_WORKERS = self.n_cpus_
        train_dataset = EPFDataset(X, y, dtype=self.dtype, N_OUTPUT=self.N_OUTPUT)
        val_dataset = EPFDataset(Xv, yv, dtype=self.dtype, N_OUTPUT=self.N_OUTPUT)

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
        test_dataset = EPFDataset(X, dtype=self.dtype, N_OUTPUT=self.N_OUTPUT)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                 num_workers=NUM_WORKERS)
        return test_loader


class EPFDataset(Dataset):
    """
    Helps constructing DataLoaders
    """
    def __init__(self, X, Y=None, dtype=torch.float32, N_OUTPUT=24):
        self.dtype = dtype
        self.N_OUTPUT = N_OUTPUT
        self.X = torch.tensor(X.astype(float), dtype = dtype)

        if Y is not None:
            self.Y = torch.tensor(Y.astype(float), dtype = dtype)
        else:
            self.Y = None
            
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.Y is not None:
            return (self.X[idx, :], self.Y[idx, :])
        else:
            return self.X[idx, :]


    

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback
import numpy as np, matplotlib, matplotlib.pyplot as plt, torch

class StoreOBhat(Callback):
    def __init__(self, dtype=torch.float32):
        Callback.__init__(self)
        self.dtype = dtype
        
        self.OBhats = [[]]
        self.current_epoch = -1        

    def on_train_epoch_start(self, trainer, pl_module):
        self.current_epoch += 1 
        self.OBhats.append([])        

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        x, y = batch
        OBhat = pl_module.forward(x, predict_order_books=True).detach().numpy()
        self.OBhats[self.current_epoch].append(OBhat)        

class StoreLosses(Callback):
    def __init__(self):
        Callback.__init__(self)
        self.current_epoch = -1
        self.current_val_epoch = -1
        
        self.train_loss_step = []
        self.train_loss_epoch = []
        
        self.val_loss = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.current_epoch += 1
        self.train_loss_step.append([])
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        value = trainer.callback_metrics.get("train_loss_step").item()
        self.train_loss_step[self.current_epoch].append(value)       

    def on_train_epoch_end(self, trainer, pl_module):
        value = trainer.callback_metrics.get("train_loss_epoch").item()
        self.train_loss_epoch.append(value)

    def on_validation_start(self, trainer, pl_module):
        self.current_val_epoch += 1
        self.val_loss.append([])

    def on_validation_end(self, trainer, pl_module):
        value = trainer.callback_metrics.get("val_loss").item()
        self.val_loss[self.current_val_epoch].append(value)

    def plot_epochs(self):
        plt.plot(self.train_loss_epoch, label="Train Loss", c="b", lw=3)
        plt.plot(np.array(self.val_loss).reshape(-1), label="Val Loss", c="r", lw=3)
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.grid("on")
        plt.show()

class EarlyStoppingSlidingAverage(EarlyStopping):
    def __init__(self, monitor="val_loss", verbose=False, alpha=10, patience=10,
                 restore_best_weights=False):
        EarlyStopping.__init__(self, monitor=monitor, verbose=verbose,
                               patience=patience)
        self.alpha = alpha
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.stopped_epoch = 0
        self.val_losses = []            
        self.best = 10e9
        self.best_weights = None
        
    def on_validation_end(self, trainer, pl_module):
        current = self.get_monitor_value(trainer.callback_metrics)
        if current is None:
            return

        if current < self.best:
            self.best = current
            self.wait = 0
            if self.restore_best_weights and verbose > 0:
                print("Can't restore weights of a torch module")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = trainer.current_epoch
                trainer.should_stop = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print("Can't restore weights of a torch module")
        
    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor).detach().numpy()
        if monitor_value is None:
            print(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        self.val_losses.append(monitor_value.ravel()[0])
            
        try:
            current = len(self.val_losses)
        except:
            current = 0
            values = [self.val_losses[current]]
        else:
            k = min(self.alpha, current)
            values = self.val_losses[current - k:current]
            
        mean_ = np.mean(values)
        return mean_

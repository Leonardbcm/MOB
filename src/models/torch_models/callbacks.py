from PIL import Image
import io
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback
import numpy as np, matplotlib, matplotlib.pyplot as plt, torch, os
from src.euphemia.ploters import get_ploter

from src.euphemia.order_books import TorchOrderBook

class PlotOBCallback(Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir

    def latest_version(self):
        """
        Get the most recent version = version with the highest number
        """
        path = self.save_dir
        all_versions = os.listdir(path)
        all_v = np.array([int(v.split("_")[1]) for v in all_versions])
        return all_versions[np.argmax(all_v)]

    # Inverse transform
    def inverse_OB(self, ob, pl_module):
        # a Compute the retrieved hour
        h = pl_module.idx_to_log % 24
        
        # Compute insertion indices
        vi = pl_module.yv_indices[h*pl_module.OBs:(h+1)*pl_module.OBs]
        poi = pl_module.ypo_indices[h*pl_module.OBs:(h+1)*pl_module.OBs]
        pi = pl_module.yp_indices[h*pl_module.OBs:(h+1)*pl_module.OBs]
        all_indices = np.concatenate((vi, poi, pi))
        
        OBfilled = np.zeros((1, pl_module.transformer.n_features_in_))
        for v, i in zip([vi, poi, pi], [0, 1, 2]):
            OBfilled[:, v] = ob[:, i]
            
        OB_inversed = pl_module.transformer.inverse_transform(
            OBfilled).round(decimals=2)
        OB_destination = np.zeros((pl_module.OBs, 3))
        for v, i in zip([vi, poi, pi], [0, 1, 2]):
            OB_destination[:, i] = OB_inversed[:, v]
            
        return OB_destination    

    def on_train_epoch_end(self, trainer, pl_module):        
        # 1. Retrieve stored data
        OBhat = pl_module.OB["hat"]
        OBskipped = pl_module.OB["skipped"]        
        OB = pl_module.OB["true"]
        OB_to_inverse = [OBhat, OBskipped, OB]        
        
        ###### 2 Inverse transform the order books
        OB_objects = [self.inverse_OB(ob, pl_module) for ob in OB_to_inverse]
        OBhat, OBskipped, OB = OB_objects            

        # Get current version
        cv = self.latest_version()
        
        # Save OBhat to the disk
        save_dir = os.path.join(self.save_dir, cv)
        print(save_dir)
        np.save(os.path.join(save_dir, f"OBhat_{trainer.current_epoch}"), OBhat)
        
        # Only works for OBhat
        OBhat = TorchOrderBook(OBhat)
        OBskipped = TorchOrderBook(OBskipped) 
        OB = TorchOrderBook(OB)
            
        # 3. Make the plot
        fig, ax = plt.subplots(1, figsize=(19.2*0.66, 10.8*0.66))
        get_ploter(OB).display(ax_=ax, labels="OB", colors="r")
        get_ploter(OBhat).display(ax_=ax, labels="OBhat", colors="b")
        get_ploter(OBskipped).display(ax_=ax, labels="OBskipped", colors="g")
        ax.legend()
        
        # Log the figure using the provided logger
        trainer.logger.experiment.add_figure(
            f"Forecasted Order Book", fig, global_step=trainer.current_epoch)

        fig.set_figwidth(19.2)
        fig.set_figheight(10.8)
        fig.savefig(os.path.join(
            save_dir, f"OrderBooks_epoch_{trainer.current_epoch}.png"))
        
        
        # Close the figure to prevent memory leaks
        plt.close(fig)        
        
            
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
                print(f"Stopping Training after wait = {self.wait}")
                print(f"{self.monitor} is at {current} while best is {self.best}")
                trainer.should_stop = True
                
        
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

class ValOBhat(Callback):
    def __init__(self, save_to_disk="", dtype=torch.float32):
        Callback.__init__(self)
        self.dtype = dtype
        self.save_to_disk = save_to_disk

        os.makedirs(self.save_to_disk, exist_ok=True)            
        
        self.current_epoch = -2
        self.current_batch = -1

    def on_validation_epoch_start(self, trainer, pl_module):
        self.current_epoch += 1
        self.current_batch = -1
        
        self.current_epoch_path = os.path.join(
            self.save_to_disk, "epoch_" + str(self.current_epoch))
        os.makedirs(self.current_epoch_path, exist_ok=True)

    def on_validation_batch_end(self,trainer, pl_module, outputs, batch, batch_idx,
                                dataloader_idx):
        self.current_batch += 1        
        x, y = batch

        # Predict what we want
        yhat = pl_module.forward(x.detach()).detach().numpy()
        V3, Po3, PoP3, P3 = pl_module.forward(x.detach(),return_parts="step_3")
        OBhat = pl_module.forward(
            x.detach(), predict_order_books=True).detach().numpy()

        # detach tensors
        P3 = P3.detach().numpy()                
        Po3 = Po3.detach().numpy()                
        
        self.current_batch_path = os.path.join(
            self.current_epoch_path, "validation_batch_" + str(self.current_batch))
        os.makedirs(self.current_batch_path, exist_ok=True)

        if self.current_batch == 0:
            np.save(os.path.join(self.current_batch_path, "OBhat.npy"), OBhat)
            np.save(os.path.join(self.current_batch_path, "yhat.npy"), yhat)    
        np.save(os.path.join(self.current_batch_path, "Po3.npy"), Po3)
        np.save(os.path.join(self.current_batch_path, "P3.npy"), P3)          

class StoreOBhat(Callback):
    def __init__(self, save_to_disk="", dtype=torch.float32):
        Callback.__init__(self)
        self.dtype = dtype
        self.save_to_disk = save_to_disk
        
        os.makedirs(self.save_to_disk, exist_ok=True)            
        
        self.current_epoch = -1
        self.current_batch = -1

    def on_train_epoch_start(self, trainer, pl_module):
        self.current_epoch += 1
        self.current_batch = -1
        
        self.current_epoch_path = os.path.join(
            self.save_to_disk, "epoch_" + str(self.current_epoch))
        os.makedirs(self.current_epoch_path, exist_ok=True)            
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.current_batch += 1        
        x, y = batch

        # Predict what we want
        OBhat = pl_module.forward(
            x.detach(), predict_order_books=True).detach().numpy()
        yhat = pl_module.forward(x.detach()).detach().numpy()
        V1, Po1, PoP1, P1 = pl_module.forward(x.detach(),return_parts="step_1")
        V2, Po2, PoP2, P2 = pl_module.forward(x.detach(),return_parts="step_2")
        V3, Po3, PoP3, P3 = pl_module.forward(x.detach(),return_parts="step_3")

        # detach tensors
        V1 = V1.detach().numpy()
        V2 = V2.detach().numpy()
        V3 = V3.detach().numpy()                
        
        P1 = P1.detach().numpy()
        P2 = P2.detach().numpy()
        P3 = P3.detach().numpy()                

        Po1 = Po1.detach().numpy()
        Po2 = Po2.detach().numpy()
        Po3 = Po3.detach().numpy()                

        PoP1 = PoP1.detach().numpy()
        PoP2 = PoP2.detach().numpy()
        PoP3 = PoP3.detach().numpy()
        
        self.current_batch_path = os.path.join(
            self.current_epoch_path, "train_batch_" + str(self.current_batch))
        os.makedirs(self.current_batch_path, exist_ok=True)
            
        np.save(os.path.join(self.current_batch_path, "OBhat.npy"), OBhat)
        np.save(os.path.join(self.current_batch_path, "yhat.npy"), yhat)
        
        np.save(os.path.join(self.current_batch_path, "V1.npy"), V1)
        np.save(os.path.join(self.current_batch_path, "Po1.npy"), Po1)
        np.save(os.path.join(self.current_batch_path, "PoP1.npy"), PoP1)
        np.save(os.path.join(self.current_batch_path, "P1.npy"), P1)
        
        np.save(os.path.join(self.current_batch_path, "V2.npy"), V2)
        np.save(os.path.join(self.current_batch_path, "Po2.npy"), Po2)
        np.save(os.path.join(self.current_batch_path, "PoP2.npy"), PoP2)
        np.save(os.path.join(self.current_batch_path, "P2.npy"), P2)
        
        np.save(os.path.join(self.current_batch_path, "V3.npy"), V3)
        np.save(os.path.join(self.current_batch_path, "Po3.npy"), Po3)
        np.save(os.path.join(self.current_batch_path, "PoP3.npy"), PoP3)
        np.save(os.path.join(self.current_batch_path, "P3.npy"), P3)       

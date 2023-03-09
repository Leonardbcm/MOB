import torch, time, numpy as np
from pytorch_lightning import LightningModule
from torch import nn

from src.models.scalers import TorchMinMaxScaler, TorchMinAbsScaler, TorchCliper
from src.models.obn.torch_solver import PFASolver, BatchPFASolver

class SolvingNetwork(LightningModule):
    def __init__(self, din, NN1, OBs, OB_input, batch_norm, criterion, N_OUTPUT, k,
                 batch_solve, niter, pmin, pmax, step, mV, check_data, transformer,
                 scale, clip):
        LightningModule.__init__(self)
        self.NN1_input = list(NN1)
        
        # Check that the number of neurons of the last layer of NN1 is reshapable
        # by 24:
        if (self.NN1_input[-1] % 24) != 0:
            raise Exception(f"The last layer of the Feature Extracter can't be reshaped to 24bs x nf : {self.NN1[-1]} % 24 != 0!")
        
        self.din = din
        self.OB_input = list(OB_input)        
        self.OBs = OBs
        self.OBs_ = 3 * OBs
        self.N_OUTPUT = N_OUTPUT
        self.k = k
        self.niter = niter
        self.batch_solve = batch_solve
        self.batch_norm = batch_norm
        self.check_data = check_data
        self.transformer = transformer

        self.pmin = pmin
        self.pmax = pmax
        self.step = step
        self.mV = mV
        self.clip = clip
        self.scale = scale

        ################ Construct the feature extracter NN1
        nn_layers = []
        for (din, dout) in zip([self.din]+self.NN1_input[:-1], self.NN1_input):
            if self.batch_norm:
                nn_layers.append(torch.nn.BatchNorm1d(din))
            nn_layers.append(torch.nn.Linear(din, dout))
            nn_layers.append(torch.nn.ReLU())
        self.NN1 = torch.nn.Sequential(*nn_layers)

        self.in_obs = int(self.NN1_input[-1] / 24)

        ############## Construct the OrderBook forecaster
        OB_layers = []
        for (din, dout) in zip([self.in_obs]+self.OB_input,
                               self.OB_input + [3 * self.OBs]):
            if self.batch_norm:
                OB_layers.append(torch.nn.BatchNorm1d(din))
            OB_layers.append(torch.nn.Linear(din, dout))

            # Don't add relu for the last layer
            if dout != 3 * self.OBs:
                OB_layers.append(torch.nn.ReLU())
            
        self.OB_layers = torch.nn.Sequential(*OB_layers)

        ############## Create the solver with the correct bounds
        self.create_solver()
        
        self.criterion = criterion
        self.criterion_ = getattr(nn, self.criterion)()        

    def create_solver(self):
        ################# Construct the Solver
        # Compute the Scaled upper and lower bounds
        self.pmin_scaled = self.transformer.transform(
            self.pmin * np.ones((1, self.N_OUTPUT))).min()
        self.pmax_scaled = self.transformer.transform(
            self.pmax * np.ones((1, self.N_OUTPUT))).max()
        self.step_scaled = (self.pmax_scaled - self.pmin_scaled) / ((self.pmin - self.pmax) / self.step)
        
        if not self.batch_solve:
            self.optim_layer = PFASolver(
                pmin=self.pmin_scaled, pmax=self.pmax_scaled, step=self.step_scaled,
                k=self.k,mV=self.mV, check_data=self.check_data)
        else:
            self.optim_layer = BatchPFASolver(
                niter=self.niter, pmin=self.pmin, pmax=self.pmax,
                k=self.k, mV=self.mV, check_data=self.check_data)        
        
    def forward(self, x, predict_order_books=False):
        ############## 1] Forecast Order Books
        # Extract features for each day
        x = self.NN1(x)

        # Reshape the data for hourly prediction
        x = x.reshape(-1, self.in_obs)

        # Forecast order books for each hour
        x =  self.OB_layers(x)

        ############## 2] Format Order Books
        x = x.reshape(-1, self.OBs, 3)
        V, Po, PoP = x[:, :, 0], x[:, :, 1], x[:, :, 2]        

        # Scale the forecsted price to the price range
        if self.scale:
            Po = MinMaxScaler(a=self.pmin_scaled, b=self.pmax_scaled)(Po)        
            PoP = MinMaxScaler(a=self.pmin_scaled, b=self.pmax_scaled)(PoP)
            
            # Scale the volumes so that they are not too small
            # V = MinAbsScaler(m=self.mV)(V)            
        elif self.clip:
            Po = Cliper(self.pmin_scaled, self.pmax_scaled)(Po)
            PoP = Cliper(self.pmin_scaled, self.pmax_scaled)(PoP)

            # Clip the volumes so that they are not too small
            # V = AbsCliper(self.mV, k=self.k)(V)

        # Compute P
        P = PoP - Po

        # Store the percentage of demand orders
        self.perc_neg = 100 * torch.mean(torch.where(P < 0, 1.0, 0.0))
        
        # Ensure that V and P have the same sign
        psigns = 2.0 * torch.sigmoid(self.k * P) - 1.0
        V = torch.abs(V) * psigns

        # Reconstruct the OB
        x = torch.cat([
            V.reshape(-1, self.OBs, 1),
            Po.reshape(-1, self.OBs, 1),
            P.reshape(-1, self.OBs, 1)], axis=2)
        
        if predict_order_books:
            return x
        
        ############## 3] Solve the problem(s)
        x = self.optim_layer(x)
        x = x.reshape(-1, self.N_OUTPUT)
        
        return x

    def _step(self, batch, batch_idx):
        x, y = batch
        xout = self.forward(x)
        loss = self.criterion_(xout.reshape_as(y), y)
        bs = int(x.shape[0])
        return loss, bs
    
    def training_step(self, data, batch_idx):
        loss, bs = self._step(data, batch_idx)
        self.log("train_loss", loss, batch_size=bs, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log("train_percentage_of_demand_orders", self.perc_neg, logger=True)
        self.log("pmin_scaled", self.pmin_scaled, logger=True)
        self.log("pmax_scaled", self.pmax_scaled, logger=True)        
        return loss

    def validation_step(self, data, batch_idx):
        loss, bs = self._step(data, batch_idx)  
        self.log("val_loss", loss, batch_size=bs, logger=True)
        self.log("val_percentage_of_demand_orders", self.perc_neg, logger=True)        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

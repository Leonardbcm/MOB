import torch, time, numpy as np
from pytorch_lightning import LightningModule
from torch import nn

from src.models.torch_models.scalers import TorchMinMaxScaler, TorchMinAbsScaler, TorchCliper
from src.models.torch_models.torch_solver import PFASolver, BatchPFASolver

class SolvingNetwork(LightningModule):
    def __init__(self, din, NN1, OBs, OB_input, batch_norm, criterion, N_OUTPUT, k,
                 batch_solve, niter, pmin, pmax, step, mV, check_data, transformer,
                 scale, weight_initializers):
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
        self.scale = scale
        self.weight_initializers = weight_initializers
        self.OB_weight_initializers = {
            "polayers" : self.weight_initializers,
            "poplayers" : self.weight_initializers}

        ################ Construct the feature extracter NN1
        nn_layers = []
        for (din, dout) in zip([self.din] + self.NN1_input[:-1], self.NN1_input):
            if self.batch_norm:
                nn_layers.append(torch.nn.BatchNorm1d(din))
            nn_layers.append(torch.nn.Linear(din, dout))
            nn_layers.append(torch.nn.ReLU())
        self.NN1 = torch.nn.Sequential(*nn_layers)

        self.in_obs = int(self.NN1_input[-1] / 24)

        ############## Construct the OrderBook forecaster if at least 1 layer
        OB_layers = []
        self.OB_out = self.in_obs
        if len(self.OB_input) > 0:
            self.OB_out = self.OB_input[-1]
            for (din, dout) in zip(
                    [self.in_obs] + self.OB_input[:-1],
                    self.OB_input):
                if self.batch_norm:
                    OB_layers.append(torch.nn.BatchNorm1d(din))
                
                OB_layers.append(torch.nn.Linear(din, dout))
                OB_layers.append(torch.nn.ReLU())
            
        self.OB_layers = torch.nn.Sequential(*OB_layers)
        
        ############## Intialize the V, Po, P layers
        containers = {"vlayers" : [], "polayers" : [], "poplayers" : []}       
        for container in containers.keys():
            c_layers = containers[container]
            if self.batch_norm:
                c_layers.append(torch.nn.BatchNorm1d(self.OB_out))

            forecast_layer = torch.nn.Linear(self.OB_out, self.OBs)
            
            # Special weight init
            if container in self.OB_weight_initializers.keys():
                inits = self.OB_weight_initializers[container]
                for init in inits:
                    # Udpate the layer intializer using the transformer
                    init.update(self.transformer)

                    # Intialize the layer
                    init(forecast_layer)
                    
            c_layers.append(forecast_layer)              
            
            seq = torch.nn.Sequential(*c_layers)
            setattr(self, container, seq)
                
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
        
    def forward(self, x, predict_order_books=False, return_parts=""):
        ############## 1] Forecast Order Books
        # Extract features for each day
        x = self.NN1(x)

        # Reshape the data for hourly prediction
        x = x.reshape(-1, self.in_obs)

        # Forecast order books for each hour
        x =  self.OB_layers(x)

        V = self.vlayers(x)
        Po = self.polayers(x)
        PoP = self.poplayers(x)
        P = PoP - Po
        
        if return_parts == "step_1":
            return V, Po, PoP, P
        
        ############## 2] Fit OB to the domain        
        if self.scale == "MinMax":
            Po = TorchMinMaxScaler(a=self.pmin_scaled, b=self.pmax_scaled)(Po)
            PoP = TorchMinMaxScaler(a=self.pmin_scaled, b=self.pmax_scaled)(PoP)
        if self.scale == "Clip":
            Po = TorchCliper(self.pmin_scaled, self.pmax_scaled)(Po)
            PoP = TorchCliper(self.pmin_scaled, self.pmax_scaled)(PoP)     
                        
        # Compute P
        P = PoP - Po
        if return_parts == "step_2":
            return V, Po, PoP, P          
            
        # Handle signs
        Ss = torch.sigmoid(self.k * P)
        signs = 2.0 * Ss - 1.0 + 4 * Ss * (1 - Ss)        
        
        # Ensure that V and P have the same sign
        V = torch.abs(V) * signs
        P = torch.abs(P) * signs

        PoP = Po + P
        if return_parts == "step_3":
            return V, Po, PoP, P           
            
        # Store the percentage of demand orders
        self.perc_neg = 100 * torch.mean(torch.where(P < 0, 1.0, 0.0))

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

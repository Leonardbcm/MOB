import torch, time, numpy as np
from pytorch_lightning import LightningModule
from torch.profiler import profile, record_function, ProfilerActivity
from torchmetrics import SymmetricMeanAbsolutePercentageError
from torch import nn

from src.models.torch_models.scalers import TorchMinMaxScaler, TorchCliper
from src.models.torch_models.sign_layer import SignLayer
from src.models.torch_models.torch_solver import PFASolver, BatchPFASolver


class SolvingNetwork(LightningModule):
    def __init__(self, din, NN1, OBs, OB_input, batch_norm, criterion, N_OUTPUT, k,
                 batch_solve, niter, pmin, pmax, step, mV, check_data, transformer,
                 OB_transformer, scale, weight_initializers, profile,
                 skip_connection, use_order_books, separate_optim, N_PRICES,
                 dropout):
        LightningModule.__init__(self)

        # Architecture parameters
        self.skip_connection = skip_connection
        self.use_order_books = use_order_books
        self.separate_optim = separate_optim        

        # Network structure parameters
        self.din = din        
        self.NN1_input = list(NN1)
        self.OB_input = list(OB_input)
        self.OBs = OBs
        self.N_OUTPUT = N_OUTPUT
        self.N_PRICES = N_PRICES
        self.dropout = dropout

        # Other network parameters
        self.transformer = transformer
        self.OB_transformer = OB_transformer                
        self.weight_initializers = weight_initializers
        self.OB_weight_initializers = {
            "polayers" : [wi.a_copy() for wi in self.weight_initializers],
            "poplayers" :  [wi.a_copy() for wi in self.weight_initializers]}
        self.scale = scale                
        self.batch_norm = batch_norm        

        # Solver parameters
        self.k = k
        self.niter = niter
        self.batch_solve = batch_solve
        self.pmin = pmin
        self.pmax = pmax
        self.step = step
        self.mV = mV        
        self.check_data = check_data

        # Training parameters
        self.criterion = criterion
        if self.criterion != "smape":
            self.criterion_ = getattr(nn, self.criterion)()
        else:
            self.criterion_ = SymmetricMeanAbsolutePercentageError()

        # Logging parameters
        self.profile = profile

        # Check that the number of neurons of the last layer of NN1 is reshapable
        # by 24:
        if (self.NN1_input[-1] % 24) != 0:
            raise Exception(f"The last layer of the Feature Extracter can't be reshaped to 24bs x nf : {self.NN1[-1]} % 24 != 0!")        

        # Update PMIN and PMAX according to the transformer
        self.update_bounds()

        # Construct network up until before the OB layers
        self.construct_network()
        
        # Construct the scalers if specified
        self.construct_scalers()

        # Intialize the sign Layer
        self.sign_layer = SignLayer(self.k, self.scale)
            
        # Intialize the V, Po, P layers
        self.construct_oblayers()
                
        # Create the solver
        self.create_solver()

        # Initialize order book indices
        self.init_indices()

    def construct_network(self):
        ################ Construct the feature extracter NN1
        nn_layers = []
        for (din, dout) in zip([self.din] + self.NN1_input[:-1], self.NN1_input):
            nn_layers.append(torch.nn.Linear(din, dout))
            if self.batch_norm:
                nn_layers.append(torch.nn.BatchNorm1d(dout))            
            if self.dropout > 0:
                nn_layers.append(torch.nn.Dropout(self.dropout))            
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
                OB_layers.append(torch.nn.Linear(din, dout))
                if self.batch_norm:
                    OB_layers.append(torch.nn.BatchNorm1d(dout))
                if self.dropout > 0:
                    OB_layers.append(torch.nn.Dropout(self.dropout))
                OB_layers.append(torch.nn.ReLU())
            
        self.OB_layers = torch.nn.Sequential(*OB_layers)        

    def construct_scalers(self):
        if self.scale == "MinMax":
            self.Po_scaler = TorchMinMaxScaler(
                a=self.pmin_scaled, b=self.pmax_scaled)
            self.PoP_scaler = TorchMinMaxScaler(
                a=self.pmin_scaled, b=self.pmax_scaled)
        elif self.scale == "Clip":
            self.Po_scaler = TorchCliper(self.pmin_scaled, self.pmax_scaled)
            self.PoP_scaler = TorchCliper(self.pmin_scaled, self.pmax_scaled)
        elif self.scale == "Clip-Sign":
            self.Po_scaler = TorchCliper(self.pmin_scaled, self.pmax_scaled)
            self.PoP_scaler = TorchCliper(self.pmin_scaled, self.pmax_scaled,
                                          max_frac=0.9, min_frac=0.9)            
        else:
            self.Po_scaler = torch.nn.Identity()
            self.PoP_scaler = torch.nn.Identity()

    def construct_oblayers(self):
        containers = {"vlayers" : [], "polayers" : [], "poplayers" : []}       
        for container in containers.keys():
            c_layers = containers[container]
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
        
    def update_bounds(self):
        # Compute the Scaled upper and lower bounds
        self.pmin_scaled = self.transformer.transform(
            self.pmin * np.ones((1, self.N_PRICES))).min()
        self.pmax_scaled = self.transformer.transform(
            self.pmax * np.ones((1, self.N_PRICES))).max()
        self.step_scaled = (self.pmax_scaled - self.pmin_scaled) / ((self.pmin - self.pmax) / self.step)        
        
    def create_solver(self):
        ################# Construct the Solver        
        if not self.batch_solve:
            self.optim_layer = PFASolver(
                pmin=self.pmin_scaled, pmax=self.pmax_scaled, step=self.step_scaled,
                k=self.k,mV=self.mV, check_data=self.check_data)
        else:
            self.optim_layer = BatchPFASolver(
                niter=self.niter, pmin=self.pmin, pmax=self.pmax,
                k=self.k, mV=self.mV, check_data=self.check_data)        

    def init_indices(self):
        if self.skip_connection or self.separate_optim:
            if self.use_order_books:
                start = self.din - self.OBs*24*3
            else:
                start = self.din
            
            self.v_indices = [start + 3*self.OBs*h+i for h in range(24)
                              for i in range(self.OBs)]
            self.po_indices = [v + self.OBs for v in self.v_indices]
            self.p_indices = [po + self.OBs for po in self.po_indices]
            
    def forward(self, x, predict_order_books=False, return_parts="",
                return_skipped=False):
        # If profiling is specified
        if self.profile:
            self.profile.step()            
            return self.forward_profiled(x)

        ############## 0] Separate OB from predictive data
        # Save Order books from the data
        # Separate them if use_order_books is False
        if self.skip_connection or self.separate_optim:
            with torch.no_grad():
                V_skipped = x[:, self.v_indices].reshape(-1, self.OBs)
                Po_skipped = x[:, self.po_indices].reshape(-1, self.OBs)
                P_skipped = x[:, self.p_indices].reshape(-1, self.OBs)
                PoP_skipped = Po_skipped + P_skipped

                if return_skipped:
                    x_skipped = torch.cat([
                        V_skipped.reshape(-1, self.OBs, 1),
                        Po_skipped.reshape(-1, self.OBs, 1),
                        P_skipped.reshape(-1, self.OBs, 1)], axis=2)
                    yhat_skipped=self.optim_layer(x_skipped).reshape(
                        -1, self.N_PRICES)
            
                if not self.use_order_books:            
                    x = x[:, :self.din]
        
        ############## 1] Forecast Order Books                
        # Extract features for each day                
        x = self.NN1(x)

        # Reshape the data for hourly prediction
        x = x.reshape(-1, self.in_obs)

        # Hourly features extractions
        x =  self.OB_layers(x)

        # Forecast elements of the order book : V, Po and P.
        V = self.vlayers(x)
        Po = self.polayers(x)
        PoP = self.poplayers(x)

        if return_parts == "step_0":
            P = PoP - Po            
            return V, Po, PoP, P               

        if self.skip_connection:
            V += V_skipped
            Po += Po_skipped
            PoP += PoP_skipped        
                    
        if return_parts == "step_1":
            P = PoP - Po            
            return V, Po, PoP, P
        
        ############## 2] Fit OB to the domain
        # a) Call the scaler, which are identity if no scalers where specified!
        Po = self.Po_scaler(Po)
        PoP = self.PoP_scaler(PoP)
        
        # b) Compute P
        P = PoP - Po
        if return_parts == "step_2":
            return V, Po, PoP, P          
    
        # c) Handle signs
        P, V = self.sign_layer(P, V)

        # () Compute PoP for distribution analysis
        PoP = Po + P
        if return_parts == "step_3":
            return V, Po, PoP, P           
            
        # () Store the percentage of demand orders for analysis
        self.perc_neg = 100 * torch.mean(torch.where(P < 0, 1.0, 0.0))

        # d) Reconstruct the OB        
        x = torch.cat([
            V.reshape(-1, self.OBs, 1),
            Po.reshape(-1, self.OBs, 1),
            P.reshape(-1, self.OBs, 1)], axis=2)

        if predict_order_books:
            return x
        
        ############## 3] Solve the problem(s)
        x = self.optim_layer(x)
        x = x.reshape(-1, self.N_PRICES)
        
        return x

    def forward_profiled(self, x):
        with record_function("NN1"):
            x = self.NN1(x)
            x = x.reshape(-1, self.in_obs)

        with record_function("OB_layers"):
            x =  self.OB_layers(x)

            V = self.vlayers(x)
            Po = self.polayers(x)
            PoP = self.poplayers(x)
            
        with record_function("OB_coerce"):            
            Po = self.Po_scaler(Po)
            PoP = self.PoP_scaler(PoP)
            P = PoP - Po

            P, V = self.sign_layer(P, V)
            x = torch.cat([
                V.reshape(-1, self.OBs, 1),
                Po.reshape(-1, self.OBs, 1),
                P.reshape(-1, self.OBs, 1)], axis=2)

        with record_function("Solver"):            
            x = self.optim_layer(x)
            x = x.reshape(-1, self.N_PRICES)
        
        return x    
    
    def training_step(self, data, batch_idx):
        x, y = data
        xout = self.forward(x, predict_order_books=self.separate_optim)        
        loss = self.criterion_(xout.reshape_as(y), y)
        bs = int(x.shape[0])
        
        self.log("train_loss", loss, batch_size=bs, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log("train_percentage_of_demand_orders", self.perc_neg, logger=True)
        self.log("pmin_scaled", self.pmin_scaled, logger=True)
        self.log("pmax_scaled", self.pmax_scaled, logger=True)        
        return loss

    def validation_step(self, data, batch_idx):
        x, y = data
        xout = self.forward(x, predict_order_books=False)
        loss = self.criterion_(xout.reshape_as(y), y)
        bs = int(x.shape[0])
        
        self.log("val_loss", loss, batch_size=bs, logger=True)
        self.log("val_percentage_of_demand_orders", self.perc_neg, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        xout = self.forward(batch, predict_order_books=False)
        return xout    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

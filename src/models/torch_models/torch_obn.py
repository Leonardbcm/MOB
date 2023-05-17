import torch, time, numpy as np
from pytorch_lightning import LightningModule
from torch.profiler import profile, record_function, ProfilerActivity
from torchmetrics import SymmetricMeanAbsolutePercentageError
from torch import nn
from io import StringIO
import pandas, re, os

from src.models.torch_models.scalers import TorchMinMaxScaler, TorchCliper
from src.models.torch_models.sign_layer import FixedSignLayer
from src.models.torch_models.torch_solver import PFASolver, BatchPFASolver
from src.analysis.evaluate import mae, smape, mape, rmse, rmae, dae, cmap, ACC

class SolvingNetwork(LightningModule):
    def __init__(self, din, NN1, OBs, OB_input, batch_norm, k,
                 batch_solve, niter, pmin, pmax, step, mV, check_data, transformer,
                 scale, weight_initializers, profile, skip_connection,
                 use_order_books, predict_order_books, alpha, beta,
                 gamma, coefs, N_PRICES, N_OUTPUT, dropout,
                 y_indices, yv_indices, ypo_indices, yp_indices,
                 x_indices, v_indices, po_indices, p_indices):
        super().__init__()
        self.transformer = transformer
            
        # Architecture parameters
        self.skip_connection = skip_connection
        self.use_order_books = use_order_books
        self.predict_order_books = predict_order_books

        # Coefficients
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.coefs = coefs

        # Network structure parameters
        self.din = din
        self.NN1_input = list(NN1)
        self.OB_input = list(OB_input)
        self.OBs = OBs
        self.N_PRICES = N_PRICES
        self.N_OUTPUT = N_OUTPUT
        self.dropout = dropout

        # Other network parameters
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
        self.criterion_ = SymmetricMeanAbsolutePercentageError()

        # Logging parameters
        self.profile = profile
        self.idx_to_log = 0
        self.OB = {}
        self.OB_is_logging = False
        self.OB_logged = False          

        # Indices
        self.y_indices = y_indices
        self.yv_indices = yv_indices
        self.ypo_indices = ypo_indices
        self.yp_indices = yp_indices
        self.yOB_indices = np.concatenate(
            (self.yv_indices, self.ypo_indices, self.yp_indices))
        
        self.x_indices = x_indices
        self.v_indices = v_indices
        self.po_indices = po_indices
        self.p_indices = p_indices        
        
        # Check that the number of neurons of the last layer of NN1 is reshapable
        # by 24:
        if (self.NN1_input[-1] % 24) != 0:
            raise Exception(f"The last layer of the Feature Extracter can't be reshaped to 24bs x nf : {self.NN1[-1]} % 24 != 0!")        

        # Update PMIN and PMAX according to the transformer
        self.update_bounds()

        ############## Common Network
        self.construct_common_network()

        ############## Alpha forecasting layer (NN alpha)
        if self.alpha > 0:
            self.construct_alpha_network()

        ############## Construct the OrderBook forecaster (NN beta)
        if (self.beta > 0) or (self.gamma > 0):
            self.construct_beta_network()        
                
        ############# Create the solver
        if self.beta > 0:
            self.create_solver()

    def construct_common_network(self):    
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

    def construct_alpha_network(self):
        self.alpha_layer = torch.nn.Linear(self.NN1_input[-1], self.N_PRICES)

    def construct_beta_network(self):
        ######### Construct the NN that forecasts OB
        if (self.beta > 0) or (self.gamma > 0):
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

        ######## Add the layers that forecasts v, po, p
        containers = {"vlayers" : [], "polayers" : [], "poplayers" : []}       
        for container in containers.keys():
            c_layers = containers[container]
            forecast_layer = torch.nn.Linear(self.OB_out, self.OBs)
            
            # Special weight init
            if (self.beta > 0) or (self.gamma > 0):
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

        ######### ADD the scaler            
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

        ######### SIGN LAYER
        self.sign_layer = FixedSignLayer(self.OBs)

    def update_bounds(self):
        if (self.beta > 0) or (self.gamma > 0):        
            # Compute the Scaled upper and lower bounds
            self.pmin_scaled = -1
            self.pmax_scaled = 1
            self.step_scaled = (self.pmax_scaled - self.pmin_scaled) / ((self.pmax - self.pmin) / self.step)

            print(f"Updated search bounds : {self.pmin}->{self.pmin_scaled},{self.pmax}->{self.pmax_scaled}")
        else:
            self.pmin_scaled = self.pmin
            self.pmax_scaled = self.pmax            
        
    def create_solver(self):
        ################# Construct the Solver        
        if not self.batch_solve:
            self.optim_layer = PFASolver(
                pmin=self.pmin_scaled, pmax=self.pmax_scaled, step=self.step_scaled,
                k=self.k,mV=self.mV, check_data=self.check_data)
        else:
            self.optim_layer = BatchPFASolver(
                niter=self.niter, pmin=self.pmin_scaled, pmax=self.pmax_scaled,
                k=self.k, mV=self.mV, check_data=self.check_data)
        
    def forward(self, x, return_order_books=False):
        ############## -1]  Init stuff
        with record_function("INIT"):
            bs = x.shape[0]        
            self.final_duals = torch.zeros(bs*24, dtype=x.dtype)
            self.perc_solved = 0.0
            self.remaining_unsolved = 0.0
            self.perc_neg = 0.0
        
            ybeta = torch.zeros((bs, self.N_PRICES), dtype=x.dtype)
            ygamma = torch.zeros((bs*24, self.OBs, 3), dtype=x.dtype)
            yalpha = torch.zeros((bs, self.N_PRICES), dtype=x.dtype)
        
        ############## 0] Separate OB from predictive data
        # Extract Order books from the data
        # Separate them if use_order_books is False
        with record_function("Skip-Connection"):        
            if self.skip_connection and ((self.gamma > 0) or (self.beta > 0)):
                with torch.no_grad():
                    V_skipped = x[:, self.v_indices].reshape(-1, self.OBs)
                    Po_skipped = x[:, self.po_indices].reshape(-1, self.OBs)
                    P_skipped = x[:, self.p_indices].reshape(-1, self.OBs)
                    PoP_skipped = Po_skipped + P_skipped

                    if not self.OB_logged:
                        OB_skipped = torch.cat([
                            V_skipped.reshape(-1, self.OBs, 1),
                            Po_skipped.reshape(-1, self.OBs, 1),
                            P_skipped.reshape(-1, self.OBs, 1)], axis=2)
                        self.OB["skipped"]=OB_skipped.detach()[self.idx_to_log,:,:]
                        
                        self.OB_is_logging = True
                        self.OB_logged = True

            # Remove order books from the data if they are not to be used!
            if not self.use_order_books:            
                x = x[:, self.x_indices]
        
        ############## I] Common NN
        with record_function("NN1"):      
            x = self.NN1(x)

        ############## II.a] Alpha model
        with record_function("ALPHA"):        
            if self.alpha > 0:
                yalpha = self.alpha_layer(x)
        
        ############## II.b] Beta model : OB forecasting
        if (self.beta > 0) or (self.gamma > 0):
            
            with record_function("OB_layers"):
            
                # 1. Forecast OB            
                # Reshape the data for hourly prediction            
                x = x.reshape(-1, self.in_obs)
                
                # Hourly features extractions : NNbeta
                x = self.OB_layers(x)
                
                # Forecast elements of the order book : V, Po and P.
                V = self.vlayers(x)
                Po = self.polayers(x)
                PoP = self.poplayers(x)

                if self.skip_connection:
                    V += V_skipped
                    Po += Po_skipped
                    PoP += PoP_skipped

            with record_function("OB_coerce"):
                    
                # 2. Fit OB to the domain
                # a) Call the scaler, which are identity if no scalers
                Po = self.Po_scaler(Po)
                PoP = self.PoP_scaler(PoP)
        
                # b) Compute P
                P = PoP - Po
                
                # c) Handle signs
                P, V = self.sign_layer(P, V)

                # () Store the percentage of demand orders for analysis
                self.perc_neg = 100 * torch.mean(torch.where(P < 0, 1.0, 0.0))
        
                # d) Reconstruct the OB        
                ygamma = torch.cat([
                    V.reshape(-1, self.OBs, 1),
                    Po.reshape(-1, self.OBs, 1),
                    P.reshape(-1, self.OBs, 1)], axis=2)

                if return_order_books:
                    return ygamma

                if self.OB_is_logging:
                    self.OB["hat"] = ygamma.detach()[self.idx_to_log, :, :]
            
            with record_function("Solver"):
                # CHECK if we have to solve the problem or not
                if self.beta > 0:                    
                    # 3] Solve the problem(s)
                    x = self.optim_layer(ygamma)
                    ybeta = x.reshape(-1, self.N_PRICES)
            
                    # Log the values of the dual problem
                    final_duals = torch.tensor(
                        self.optim_layer.DMs[self.optim_layer.steps-1,:])
                    self.final_duals = final_duals
                    self.perc_solved = 100 * torch.mean(torch.where(
                        torch.abs(self.final_duals) < 0.01, 1.0, 0.0))        
                    self.remaining_unsolved = torch.sum(torch.abs(self.final_duals))
        
        return yalpha, ygamma, ybeta

    def loss(self, yalpha, ygamma, ybeta, y, dataset="train"):
        Lalpha, Lbeta, Lgamma = 0.0, 0.0, 0.0
        # Compute ALPHA and BETA losses
        if not self.predict_order_books:
            yprices = y[:, self.y_indices]

            if self.alpha > 0:
                with record_function(f"ALPHA_LOSS_{dataset}"):
                    Lalpha = self.criterion_(yalpha.reshape_as(yprices), yprices)
            if self.beta > 0:
                with record_function(f"BETA_LOSS_{dataset}"):
                    Lbeta = self.criterion_(ybeta.reshape_as(yprices), yprices)
        losses = {"alpha" : Lalpha, "beta" : Lbeta}
        
        with record_function(f"GAMMA_LOSS_{dataset}"):        
            if self.gamma > 0:
                # Split real OB
                V = y[:, self.yv_indices].reshape(-1, self.OBs)
                Po = y[:, self.ypo_indices].reshape(-1, self.OBs)
                P = y[:, self.yp_indices].reshape(-1, self.OBs)
        
                if self.OB_is_logging:
                    OB = torch.cat([
                        V.reshape(-1, self.OBs, 1),
                        Po.reshape(-1, self.OBs, 1),
                        P.reshape(-1, self.OBs, 1)], axis=2)
                    self.OB["true"] = OB.detach()[self.idx_to_log, :, :]
                    self.OB_is_logging = False
                
                # Split OB forecasts
                Vhat = ygamma[:, :, 0]
                Pohat = ygamma[:, :, 1]
                Phat = ygamma[:, :, 2]

                # Compute losses for each part
                Lv = self.criterion_(V, Vhat)
                Lpo = self.criterion_(Po, Pohat)
                Lp = self.criterion_(P, Phat)
            
                # Combine the losses
                Lgamma = (Lv*self.coefs[0] + Lpo*self.coefs[1] + Lp*self.coefs[2]) / (self.coefs[0] + self.coefs[1] + self.coefs[2])
            
                losses["V"] = Lv
                losses["Po"] = Lpo
                losses["P"] = Lp
                
        losses["gamma"] = Lgamma
        loss = (self.alpha*Lalpha+self.beta*Lbeta+self.gamma*Lgamma) / (self.alpha+self.beta + self.gamma)
        losses["main"] = loss
        
        # Log all losses
        bs = int(y.shape[0])
        for loss in losses.keys():
            self.log(f"{dataset}_{loss}_loss", losses[loss], batch_size=bs,
                     on_epoch=True, logger=True)
        return losses["main"]

    def on_train_epoch_start(self):
        self.OB = {}
        self.OB_is_logging = False
        self.OB_logged = False

    def compute_unscaled_loss(self, yalpha, ybeta, ygamma, y, dataset, bs):
        with torch.no_grad():        
            yhat = self.format_prediction(yalpha, ybeta, ygamma, bs)
        
            # Unscale the prediction
            yhat_unscaled = self.transformer.inverse_transform(yhat)

            # Unscale the labels
            y_unscaled = self.transformer.inverse_transform(y)

            # Compute the metrics we want
            losses = {}
            if not self.predict_order_books:
                losses["price_smape"] = smape(
                    y_unscaled[:, self.y_indices], yhat_unscaled[:, self.y_indices])
                losses["price_mae"] = mae(
                    y_unscaled[:, self.y_indices], yhat_unscaled[:, self.y_indices])
                losses["price_ACC"] = ACC(
                    y_unscaled[:, self.y_indices], yhat_unscaled[:, self.y_indices])
            if self.gamma > 0:
                losses["OB_smape"] = smape(
                    y_unscaled[:, self.yOB_indices],
                    yhat_unscaled[:, self.yOB_indices])
                losses["OB_ACC"] = ACC(
                    y_unscaled[:, self.yOB_indices],
                    yhat_unscaled[:, self.yOB_indices])
            
            for loss in losses.keys():
                self.log(f"{dataset}_{loss}_loss", losses[loss], batch_size=bs,
                         on_epoch=True, logger=True)
        
    def training_step(self, data, batch_idx):
        x, y = data
        bs = int(x.shape[0])
        yalpha, ygamma, ybeta = self.forward(x)
        self.compute_unscaled_loss(yalpha, ybeta, ygamma, y, "train-unscaled", bs)
        
        # Compute the back-propagation loss
        loss = self.loss(yalpha, ygamma, ybeta, y, dataset="train")        
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.profile:
            self.profile.step()
        
    def validation_step(self, data, batch_idx):
        x, y = data
        bs = int(x.shape[0])        
        yalpha, ygamma, ybeta = self.forward(x)
        self.compute_unscaled_loss(yalpha, ybeta,ygamma,y,"validation-unscaled", bs)
        loss = self.loss(yalpha, ygamma, ybeta, y, dataset="validation")        
        return loss

    def log_values(self, loss, bs, dataset=""):
        self.log(f"{dataset}_loss", loss, batch_size=bs, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log(f"{dataset}_percentage_of_demand_orders",self.perc_neg,logger=True)
        self.log(f"{dataset}_unsolved_problems", self.remaining_unsolved,
                 logger=True)        
        self.log(f"{dataset}_percentage_of_solved_problems", self.perc_solved,
                 logger=True)
        
        return loss    

    def on_predict_start(self):
        self.duals = []

    def format_prediction(self, yalpha, ybeta, ygamma, bs):
        """
        Given yalpha, ygamma, ybeta, format the prediction and return it.
        Formatting consist in reshaping the OB and merging it with the
        price predictions.

        Also convert tensors into nupy arrays
        """
        if self.gamma > 0:
            ygamma = ygamma.detach().numpy().reshape(bs,24*self.OBs,3).reshape(
                bs,72*self.OBs,order='F')
            
        if self.predict_order_books:
            return ygamma
        else:
            # Compute combination            
            ypred = ((self.alpha*yalpha+self.beta * ybeta) / (self.alpha + self.beta)).detach().numpy()
            if self.gamma > 0:
                yhat = np.zeros((ypred.shape[0], self.N_OUTPUT))
                yhat[:, self.y_indices] = ypred
                yhat[:, np.concatenate(
                    (self.yv_indices, self.ypo_indices, self.yp_indices))] = ygamma
                return yhat
            else:
                return ypred        
    
    def predict_step(self, batch, batch_idx):
        """
        Outputs the predicted labels for a given step.
        If predict_order_books (only gamma > 0), this is an order book forecast.

        Otherwise its a linear combination of beta and alpha forecasts, and the 
        forecasted order books if gamma > 0
        """
        yalpha, ygamma, ybeta = self.forward(batch)
        self.duals += list(self.final_duals.numpy())

        bs = batch.shape[0]
        yhat = self.format_prediction(yalpha, ybeta, ygamma, bs)
        return yhat

    def predict_step_OB(self, batch, batch_idx):
        """
        Outputs the predicted order books for this step.
        OBhat are ygamma if gamma > 0 or beta > 0.
        This raises an error if gamma <= 0 AND beta <= 0

        To compute ygamma, this calls the forward function
        """
        if (self.gamma <= 0) and (self.beta <= 0):
            raise Exception("No OB forecasts available for the input parameters!")
        
        ygamma = self.forward(batch, return_order_books=True).detach().numpy()
        bs = batch.shape[0]
        return ygamma.reshape(bs,24*self.OBs, 3).reshape(bs, 72*self.OBs, order='F')
        
    def on_predict_end(self):
        self.duals = np.array(self.duals).reshape(-1, 24)

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        with record_function("BACKWARD"):
            super().backward(loss,optimizer, optimizer_idx, *args, **kwargs)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


def filter_key_averages(key_averages):
    custom_labels = ["INIT", "Skip-Connection", "NN1", "ALPHA", "OB_layers",
                     "OB_coerce","Solver", "ALPHA_UNUSED",
                     "ALPHA_LOSS_train","BETA_LOSS_train","GAMMA_LOSS_train",
                     "ALPHA_LOSS_validation","BETA_LOSS_validation",
                     "GAMMA_LOSS_validation","BACKWARD"]

    total = key_averages[0]
    filtered_events = [total] + [event for event in key_averages
                       if event.key in custom_labels]
    
    total_cpu_time = total.cpu_time_total
    data = [
        {
            "Name": event.key,
            "CPU_time_total": event.cpu_time_total,
            "CPU_time_total_percentage": 100*event.cpu_time_total/total_cpu_time,
            "Number_of_Calls": event.count
        }
        for event in filtered_events]
    
    df = pandas.DataFrame(data)    
    return df
    
def parse_key_averages_output(output):
    lines = output.strip().split("\n")
    headers_line = lines[0].strip()
    
    # Define column boundaries
    column_boundaries = [0]
    for m in re.finditer(r'\s{2,}', headers_line):
        column_boundaries.append(m.end())

    # Extract headers
    headers = [headers_line[start:end].strip() for start, end in zip(column_boundaries[:-1], column_boundaries[1:])]

    data = []
    for line in lines[2:]:
        row_data = [line[start:end].strip() for start, end in zip(column_boundaries[:-1], column_boundaries[1:])]
        row_dict = {headers[i]: row_data[i] for i in range(len(headers))}
        data.append(row_dict)

    return data


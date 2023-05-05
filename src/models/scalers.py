import numpy as np, copy, torch, warnings
from scipy import stats
from torch import nn

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from src.models.model_utils import flatten_dict


class DataScaler(TransformerMixin, BaseEstimator):
    """
    A Wrapper for several data scaling techniques.
    This handles the data spliting using a spliter.
    
    The scaling method has to be given as a string.
    """
    def __init__(self, scaling, spliter=None, GNN="",
                 psd_idx=None, country_idx=None, edges_columns_idx=None):
        self.scaling = scaling
        self.spliter = spliter
        self.GNN = GNN
        self.psd_idx = psd_idx
        self.country_idx = country_idx
        self.edges_columns_idx = edges_columns_idx
        self.is_fitted_ = False

        if GNN != "":
            self.scaler = GNNScaler(scaling, psd_idx, country_idx,
                                    edges_columns_idx, GNN)
        else:
            if self.scaling == "BCM":
                self.scaler = BCMScaler()
            if self.scaling == "Standard":
                self.scaler = StandardScaler()
            if self.scaling == "Median":
                self.scaler = MedianScaler()
            if self.scaling == "SinMedian":
                self.scaler = SinMedianScaler()
            if self.scaling == "InvMinMax":
                self.scaler = InvMinMaxScaler()
            if self.scaling == "MinMax":
                self.scaler = MinMaxScaler()
            if self.scaling not in ("BCM", "Standard", "", "Median",
                                    "SinMedian", "InvMinMax", "MinMax"):
                raise ValueError(f'Scaling parameter must be one of "BCM", "Standard", "", Median, SinMedian, InvMinMax, MinMax, not {self.scaling}')
            
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        
        if self.spliter is not None:
            (X, _) = self.spliter(X)
            
        if not self.scaling == "":
            self.scaler.fit(X)
        
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_in_')
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        if self.scaling == "": return X
        else: return self.scaler.transform(X)

    def inverse_transform(self, X, y=None):
        check_is_fitted(self, 'n_features_in_')
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        if self.scaling == "": return X
        else: return self.scaler.inverse_transform(X)


class BCMScaler(TransformerMixin, BaseEstimator):
    """
    Standardize the data using a Standard Scaler, then apply the arcsinh.
    """
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        self.n_features_in_ = X.shape[1]
        self.scaler.fit(X)        
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_in_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = self.scaler.transform(X)
        transformed_data = np.arcsinh(transformed_data)
        return transformed_data

    def inverse_transform(self, X, y=None):
        check_is_fitted(self, 'n_features_in_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        transformed_data = np.sinh(np.float128(X))        
        transformed_data = np.float32(
            self.scaler.inverse_transform(transformed_data))

        # Need to post-treat infinity in case....
        inf_idx = np.where(transformed_data == np.inf)[0]
        if len(inf_idx) > 0:
            warnings.warn("Infinity in the output!!!!!")        
        transformed_data[inf_idx] = np.sinh(80)

        return transformed_data


class MedianScaler(TransformerMixin, BaseEstimator):
    """
    Standardize the data using a Median Scaler
    """
    def __init__(self, epsilon=10e-5):
        self.epsilon = epsilon

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        self.n_features_in_ = X.shape[1]

        self.median = np.median(X, axis=0)
        self.mad = stats.median_abs_deviation(X, axis=0)
        self.mad = np.clip(self.mad, a_min=self.epsilon, a_max=None)
        
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_in_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = (X - self.median) / self.mad
        return transformed_data

    def inverse_transform(self, X, y=None):
        check_is_fitted(self, 'n_features_in_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        transformed_data = (X * self.mad) + self.median        
        return transformed_data


class SinMedianScaler(TransformerMixin, BaseEstimator):
    """
    Standardize the data using a Standard Scaler, then apply the arcsinh.
    """
    def __init__(self):
        self.scaler = MedianScaler()

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        self.n_features_in_ = X.shape[1]
        self.scaler.fit(X)        
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_in_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = self.scaler.transform(X)
        transformed_data = np.arcsinh(transformed_data)
        return transformed_data

    def inverse_transform(self, X, y=None):
        check_is_fitted(self, 'n_features_in_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        transformed_data = np.sinh(np.float128(X))        
        transformed_data = np.float32(
            self.scaler.inverse_transform(transformed_data))

        # Need to post-treat infinity in case....
        inf_idx = np.where(transformed_data == np.inf)[0]
        if len(inf_idx) > 0:
            warnings.warn("Infinity in the output!!!!!")
        transformed_data[inf_idx] = np.sinh(80)
        
        return transformed_data

class InvMinMaxScaler(TransformerMixin, BaseEstimator):
    """
    Highest value is mapped to 0 and lowest to 1
    """
    def __init__(self, epsilon=10e-5):
        self.scaler = MinMaxScaler()
        self.epsilon = epsilon

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        self.n_features_in_ = X.shape[1]

        transformed_data = 1 / np.clip(X, a_min=self.epsilon, a_max=None)
        self.scaler.fit(transformed_data)
        
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_in_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = 1 / np.clip(X, a_min=self.epsilon, a_max=None)
        transformed_data = self.scaler.transform(transformed_data)
        
        return transformed_data

    def inverse_transform(self, X, y=None):
        check_is_fitted(self, 'n_features_in_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = self.scaler.inverse_transform(transformed_data)
        transformed_data = 1 / np.clip(transformed_data, a_min=self.epsilon, a_max=None)

        return transformed_data

class ZeroMinMaxScaler(TransformerMixin, BaseEstimator):
    """
    A Min Max Scaler where 0s are excluded from the min computation.
    """
    def __init__(self, min_value=0.1, default_value=0.5, epsilon=10e-5):
        self.epsilon = epsilon
        self.min_value = min_value
        self.default_value = default_value

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        self.n_features_in_ = X.shape[1]
        self.mins_ = np.zeros(self.n_features_in_)
        self.maxs_ = np.zeros(self.n_features_in_)
        
        for i in range(self.n_features_in_):
            ind_pos = np.where(X[:, i] > 0)[0]
            data = X[ind_pos, i]

            if len(data) > 0:
                self.mins_[i] = data.min()
                self.maxs_[i] = data.max()
            
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_in_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = copy.deepcopy(X).astype(object)
        for i in range(self.n_features_in_):
            ind_pos = np.where(X[:, i] > 0)[0]
            data = X[ind_pos, i]
            # If we found non-zero data
            if len(data) > 0:
                if self.maxs_[i] > 0:
                    transformed_data[ind_pos, i] = np.clip(
                        (data - self.mins_[i]) / (
                            self.maxs_[i] - self.mins_[i] + self.epsilon),
                        self.min_value, 1)                    
                else:
                    # If the recorded max was 0 but non zero data arrives
                    transformed_data[ind_pos, i] =  self.default_value * np.ones(
                        len(ind_pos))
                    
        return transformed_data

    def inverse_transform(self, X, y=None):
        raise ValueError("This is for ATC only so no inverse transform is required")

    
class GNNScaler(TransformerMixin, BaseEstimator):
    """
    Standardizer for the GNN data.
    This scaler first leave the date column untouched.
    Then, it separates the Nodes data from the edges data.
    The nodes data is normalized using the node_scaler.
    The edge data is normalized using the edge_scaler which is a ZeroMinMaxScaler
    """
    def __init__(self, nodes_scaler, psd_idx, node_columns_idx,
                 edges_columns_idx, GNN):
        self.nodes_scaler = DataScaler(nodes_scaler, spliter=None)
        self.edges_scaler = ZeroMinMaxScaler()
        self.psd_idx = psd_idx
        self.node_columns_idx = flatten_dict(node_columns_idx)
        self.edges_columns_idx = flatten_dict(edges_columns_idx)
        self.GNN = GNN
        
    def split_node_edge(self, X):        
        dates = X[:, self.psd_idx]
        Xe = X[:, self.edges_columns_idx]
        Xn = X[:, self.node_columns_idx]    
        return dates, Xe, Xn

    def merge_node_edge(self, dates, Xe, Xn):
        X = np.empty((len(dates), Xe.shape[1] + Xn.shape[1] + 1), dtype='object')
        X[:, self.psd_idx] = dates
        X[:, self.node_columns_idx] = Xn
        X[:, self.edges_columns_idx] = Xe
        return X
    
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        dates, Xe, Xn = self.split_node_edge(X)

        self.zero_edges = np.where(Xe.mean(axis=0) == 0)[0]
        self.zero_nodes = np.where(Xn.mean(axis=0) == 0)[0]
        
        self.nodes_scaler.fit(Xn)
        if Xe.shape[1] != 0: self.edges_scaler.fit(Xe)
            
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_in_')      
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        dates, Xe, Xn = self.split_node_edge(X)        
        Xn = self.nodes_scaler.transform(Xn)
        
        if Xe.shape[1] != 0:        
            Xe = self.edges_scaler.transform(Xe)

        # Refill with 0s
        Xe[:, self.zero_edges] = 0
        Xn[:, self.zero_nodes] = 0        
            
        # Merge date, transformed_nodes and transformed_edges
        X = self.merge_node_edge(dates, Xe, Xn)
        if self.GNN == "drop_date":
            X = copy.deepcopy(X)[:, np.sort(np.concatenate(
                (self.edges_columns_idx, self.node_columns_idx)))]
            X = X.astype(np.float64)
            
        return X

    def inverse_transform(self, X, y=None):
        raise("Can't use a GNNScaler to unscale!")


class OBPriceScaler(TransformerMixin, BaseEstimator):
    """
    Scaler for Po and P part of the OB. This takes as an input the scaler that was
    used to scale the prices. With it, one shall specify the indices of the prices
    in the input data in the scaler.

    mask shall be a 1D array whose size fits the input price scaler, containing 
    booleans. The true values indicates where are the prices in the input.
    """
    def __init__(self, price_scaler, mask):
        self.price_scaler = price_scaler
        self.mask = mask
        self.is_fitted_ = False        
        
    def fill_input(self, X):
        """
        Create an input Xt which shall be compatible with this scaler's price_scaler
        The prices that have to be scaled shall be at the postion 'mask' in this 
        input.
        """
        Xt = np.zeros((X.shape[0], len(self.mask)))
        Xt[:, self.mask] = X
        return Xt
        
    def fit(self, X, y=None):
        """
        The price scaler shall be already fit before this fit function!
        """
        self.n_features_in_ = X.shape[1]

        # Check that the price scaler is fitted!
        check_is_fitted(self.price_scaler, 'n_features_in_')
        
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_in_')      
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        Xt = self.fill_input(X)
        Xt = self.price_scaler.transform(Xt)
                
        return Xt[:, self.mask]

    def inverse_transform(self, X, y=None):
        check_is_fitted(self, 'n_features_in_')      
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        Xt = self.fill_input(X)
        Xt = self.price_scaler.inverse_transform(Xt)
                
        return Xt[:, self.mask]

    
class OBNScaler(TransformerMixin, BaseEstimator):
    """
    Scaler for the OBN. The input should be :
    X = X, OB
    With 
    OB = V, Po, P
    
    This will first fit the scaler using X. Then, another scaler is used for V.
    Po and P are scaled using the non-OB prices part of the X scaler.

    The following must be provided : 
    - indices : the indices of the non-OB prices part in the non OB data
    - N_INPUT : the number of columns in the data
    - N_DATA : the number of non-OB data columns
    - x_indices : the indices of the non-OB data part
    - v_indice : the indices of the volumes of the OB
    - po_indice : the indices of PO of the OB
    - p_indice : the indices of the prices of the OB
    """
    def __init__(self, scaler_, volume_scaler_, OBs, indices, N_INPUT, N_DATA,
                 x_indices, v_indices, po_indices, p_indices, spliter=None, nh=24):

        # Spliter
        self.spliter = spliter
        
        # Data and volume scalers
        self.scaler_ = scaler_
        self.volume_scaler_ = volume_scaler_
        
        self.scaler = DataScaler(scaler_, spliter=None)
        self.volume_scaler = DataScaler(volume_scaler_, spliter=None)

        # OB Price scaler
        self.OBs = OBs
        self.indices = indices
        self.N_INPUT = N_INPUT
        self.N_DATA = N_DATA        
        self.mask = np.zeros(self.N_DATA, dtype=bool)
        self.mask[self.indices] = True
        self.price_scaler = OBPriceScaler(self.scaler, self.mask)

        # Store indices
        self.v_indices = v_indices
        self.po_indices = po_indices
        self.p_indices = p_indices
        self.x_indices = x_indices

        self.is_fitted_ = False
        self.nh = nh

    def split_OB(self, X):
        """
        Given a data matrix, split it into 4 components:
        The tabular data X
        The order book data : V, Po, P

        Po and P are reshaped to the size 24bs X OBs
        """
        V = X[:, self.v_indices]
        Po = X[:, self.po_indices]
        P = X[:, self.p_indices]        
        X = X[:, self.x_indices]

        # Reshape Po and P to shape 24*bs X OBs
        if V.shape[1] > 0:
            nx = P.shape[0]
            Por = np.zeros((nx * self.OBs, self.nh))
            Pr = np.zeros((nx * self.OBs, self.nh))
            for h in range(self.nh):
                Por[:, h] = Po[:, self.OBs*h:self.OBs*(h+1)].reshape(-1)
                Pr[:, h] = P[:, self.OBs*h:self.OBs*(h+1)].reshape(-1)
                
            Po = Por
            P = Pr
        
        return X, V, Po, P

    def reconstruct_OB(self, X, V, Po, P):
        """
        Given the components of the tabular data, reform it.
        First reshape Po and P to bs X 24OBs
        
        Then merge the data.
        """
        nx = X.shape[0]
        Porec = np.zeros((nx, self.OBs*self.nh))
        Prec = np.zeros((nx, self.OBs*self.nh))        
        for h in range(self.nh):
            Porec[:, self.OBs*h:self.OBs*(h+1)] = Po[:, h].reshape(nx, self.OBs)
            Prec[:, self.OBs*h:self.OBs*(h+1)] = P[:, h].reshape(nx, self.OBs)
        
        Xt = np.zeros((nx, self.N_INPUT))                      
        Xt[:, self.v_indices] = V
        Xt[:, self.po_indices] = Porec
        Xt[:, self.p_indices] = Prec
        Xt[:, self.x_indices] = X        
        return Xt
    
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]

        if self.spliter is not None:
            (X, _) = self.spliter(X)
            
        x, V, Po, P = self.split_OB(X)
        
        self.scaler.fit(x)    
        if V.shape[1] > 0:
            self.volume_scaler.fit(V)
            self.price_scaler.fit(Po)      
        
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_in_')      
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        X, V, Po, P = self.split_OB(X)
        X = self.scaler.transform(X)

        # Transform only if OB is not null
        if V.shape[1] > 0:        
            V = self.volume_scaler.transform(V)
            Po = self.price_scaler.transform(Po)
            P = self.price_scaler.transform(P)
            X = self.reconstruct_OB(X, V, Po, P)
            
        return X

    def inverse_transform(self, X):
        check_is_fitted(self, 'n_features_in_')
        pad = X.shape[1] == self.n_features_in_ - self.N_DATA            
        if X.shape[1] != self.n_features_in_:
            if pad:
                # In this case, the scaler has seen 72 * OBs + 24 features but
                # receives only 72 * OBs features. The real prices where only used
                # for fitting the price scaler and are not forecasted. They are
                # in place filled with 0 to inverse transform, then removed.
                OB_indices = np.concatenate((
                    self.v_indices, self.po_indices, self.p_indices))
                Xfilled = np.zeros((X.shape[0], self.n_features_in_))
                Xfilled[:, OB_indices] = X
                X = Xfilled
            else:
                raise ValueError('Shape of input is different from what was seen'
                                 'in `fit`')            

        X, V, Po, P = self.split_OB(X)
        X = self.scaler.inverse_transform(X)

        # Inverse Transform only if OB is not null
        if V.shape[1] > 0:           
            V = self.volume_scaler.inverse_transform(V)
            Po = self.price_scaler.inverse_transform(Po)
            P = self.price_scaler.inverse_transform(P)
            X = self.reconstruct_OB(X, V, Po, P)

        # Remove the padded 0s
        if pad:
            X = X[:, OB_indices]
            
        return X


class SignOBNScaler(TransformerMixin, BaseEstimator):
    """
    Scaler for the OBN. The input should be :
    X = X, OB
    
    Scales the labels of an OBN. This will split the given Y into prices and OB
    accordingly. 
    The scaler_ is fit normally. The OB scaler works differently : its a
    MinMaxScaler that maps data to the [0, 1] range and apply a given sign mask.
    This is done to ensure that the scaled Orders stays on the same direction!

    The following must be provided : 
    - scaler_ : the scaler that will be used to scale the data part
    - N_INPUT : the number of columns in the data
    - N_DATA : the number of non-OB data columns
    - x_indices : the indices of the non-OB data part
    - ob_indices : the indices of the OB part of the data
    - OBs : size of the order book
    """
    def __init__(self, scaler_, OBs, N_X, N_DATA, x_indices, v_indices,
                 po_indices, p_indices, spliter=None, nh=24):

        # Spliter
        self.spliter = spliter
        
        # Data and volume scalers
        self.scaler_ = scaler_
        self.scaler = DataScaler(scaler_, spliter=None)
        self.OB_scaler = MinMaxScaler()
        
        # OB Price scaler
        self.OBs = OBs
        self.N_X = N_X
        self.N_DATA = N_DATA

        # Create the mask 
        self.mask_unit = np.ones(self.OBs , dtype=int)
        self.mask_unit[int(self.OBs/2):] = -1
        self.mask = np.array([self.mask_unit for h in range(nh)]).reshape(-1)

        # Store indices
        self.x_indices = x_indices
        self.v_indices = v_indices
        self.po_indices = po_indices
        self.p_indices = p_indices
        self.OB_indices = np.array([i for i in np.arange(self.N_X)
                                    if i not in self.x_indices], dtype=int)
        
        self.is_fitted_ = False
        self.nh = nh

    def split_OB(self, X):
        """
        Given a data matrix, split it into 4 components:
        The tabular data X
        The order book data : OB = V, Po, P
        """
        OB = X[:, self.OB_indices]
        X = X[:, self.x_indices]
        
        return X, OB

    def reconstruct_OB(self, X, OB):
        """
        Given the components of the tabular data, reform it.
        """
        nx = X.shape[0]
        Xt = np.zeros((nx, self.N_X))                      
        Xt[:, self.OB_indices] = OB
        Xt[:, self.x_indices] = X
      
        return Xt
    
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]

        if self.spliter is not None:
            (X, _) = self.spliter(X)
            
        x, OB = self.split_OB(X)
        
        self.scaler.fit(x)    
        if OB.shape[1] > 0:
            self.OB_scaler.fit(OB)
        
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_in_')      
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        X, OB = self.split_OB(X)
        X = self.scaler.transform(X)
        
        # Transform only if OB is not null
        if OB.shape[1] > 0:        
            OB = self.OB_scaler.transform(OB)
            X = self.reconstruct_OB(X, OB)            
            X[:, self.v_indices] = self.mask * np.abs(X[:, self.v_indices])
            X[:, self.p_indices] = self.mask * np.abs(X[:, self.p_indices])
                        
        return X

    def inverse_transform(self, X):
        check_is_fitted(self, 'n_features_in_')
        pad = X.shape[1] == self.n_features_in_ - self.N_DATA            
        if X.shape[1] != self.n_features_in_:
            if pad:
                # In this case, the scaler has seen 72 * OBs + 24 features but
                # receives only 72 * OBs features. The real prices where only used
                # for fitting the price scaler and are not forecasted. They are
                # in place filled with 0 to inverse transform, then removed.
                Xfilled = np.zeros((X.shape[0], self.n_features_in_))
                Xfilled[:, self.OB_indices] = X
                X = Xfilled
            else:
                raise ValueError('Shape of input is different from what was seen'
                                 'in `fit`')            

        X, OB = self.split_OB(X)
        X = self.scaler.inverse_transform(X)

        # Inverse Transform only if OB is not null
        if OB.shape[1] > 0:           
            OB = self.OB_scaler.inverse_transform(np.abs(OB))
            X = self.reconstruct_OB(X, OB)            
            X[:, self.v_indices] = self.mask * np.abs(X[:, self.v_indices])
            X[:, self.p_indices] = self.mask * np.abs(X[:, self.p_indices])

        # Remove the padded 0s
        if pad:
            X = X[:, self.OB_indices]
            
        return X

    

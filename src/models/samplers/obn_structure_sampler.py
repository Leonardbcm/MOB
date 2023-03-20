import scipy.stats as stats, copy
from scipy.stats import rv_continuous
from src.models.samplers.structure_sampler import structure_sampler
from src.models.samplers.cnn_structure_sampler import flatten_dict


class obn_structure_sampler(rv_continuous):
    """
    Sampler for a OBN structure.
    An OBN structure has the following components:
    NN1 -> RESHAPE -> OB_layers -> (3 * OBs)

    NN1 is the feature extraction part that computes features from daily data.
    Features are then reshaped to (24*batch_size X NN1out/24)
    The OB_layers computes hourly features
    It is folowed by 3 layers of size (OBs) that forecasts V, Po, P.

    This sampler goes backward : it first samples the n_OB_layers OB_layers between
    (OB_min and OB_max) neurons. The first sampled data is then mutliplied by 24
    and becomes NN1_min. 
    The samplers then samples n_NN1_layers NN1_layers between (IN and NN1_min), 
    sorted by decreasing order.
    
    This samplers DOES NOT take care of OBs!
    """
    def __init__(self, IN, n_NN1_layers, n_OB_layers, ob_min, ob_max):
        rv_continuous.__init__(self)
        self.IN = IN
        self.n_NN1_layers = n_NN1_layers - 1
        self.n_OB_layers  = n_OB_layers + 1
        self.ob_min = ob_min
        self.ob_max = ob_max
        
        self.OB_sampler = structure_sampler(
            self.n_OB_layers, min_=self.ob_min, max_=self.ob_max, sort=False)

    def rvs(self, size=1, random_state=None):
        res = {"NN1" : [], "OBN" :  []}
        for i in range(size):            
            # Sample the OB_layers
            ob_layers = self.OB_sampler.rvs(size=1, random_state=random_state)[0]
            
            # Create the NN1 sampler using the first OB layer number of neurons * 24
            NN1_min = ob_layers[0] * 24
            min_ = min(NN1_min, self.IN)
            max_ = max(NN1_min, self.IN)
            
            self.NN1_sampler = structure_sampler(
                self.n_NN1_layers, min_=min_, max_=max_, sort=False)
            nn1_layers = self.NN1_sampler.rvs(size=1, random_state=random_state)[0]

            # Add NN1_min * 24 to the NN1_layers
            nn1_layers = tuple(list(nn1_layers) + [NN1_min])
            res["NN1"].append(nn1_layers)

            # Remove NN1_min from the OB_layers
            ob_layers = ob_layers[1:]
            res["OBN"].append(ob_layers)
            
        # Shall return a dict of arrays (of tuples!) if size > 1, only a dict
        # otherwise!
        if size > 1:
            return res

        for k in res.keys():
            res[k] = res[k][0]

        return res
            
            
            

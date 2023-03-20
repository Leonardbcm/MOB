import torch, numpy as np

#def Initializer_from_string(s):
    
class Initializer(object):
    """
    Contains parameters necessary for calling the right init function in the last
    OB layer.
    fname  is the name of the init function to use
    attribute is the attribute to initialize (weight or bias)
    p1 and p2 are the paramters (a, b) or (mean, std)
    """
    def __init__(self, fname, attribute, p1, p2, scale_infeatures=False):
        if attribute not in ("weight", "bias"):
            raise Exception("Attribute should be either weight or bias, not",
                            attribute)
        if fname not in ("normal", "uniform"):
            raise Exception("Fname should be either normal or uniform, not", fname)
        
        self.fname = fname        
        self.attribute = attribute    
        self.p1 = p1
        self.p2 = p2
        self.scale_infeatures = scale_infeatures
        
    def __call__(self, layer):
        """
        Calls the init function on attribute of the layer:
        layer.(self.attribute).data.(self.fname)_(p1, p2)
        """
        data = getattr(layer, self.attribute).data
        
        fanin = layer.in_features
        fanout = layer.out_features
        
        if self.fname == "normal":
            data.normal_(mean=self.p1, std=self.p2)            
        if self.fname == "uniform":
            data.uniform_(a=self.p1, b=self.p2)

    def update(self, transformer):
        """
        Transform attributes of this object using the given transformer.
        """
        if transformer.scaling != '':
            self.p1 = 0
            self.p2 = 1

    def __str__(self):        
        s = "Initializer("
        if self.fname == "normal":
            fnamestr = "N("
        if self.fname == "uniform":
            fnamestr = "U("
        s += f"{fnamestr}{self.p1}, {self.p2}))"
        return s
        
class BiasInitializer(Initializer):
    """
    Initialize the bias for OB price forecasting layers. The first element will 
    be set to pmin and the last to pmax. IF used to scale both Po and Po + P, this 
    will create 2 steps orders per hour, 1 supply and 1 demand.
    """    
    def __init__(self, fname, p1, p2, pmin, pmax, scale_infeatures=False):
        Initializer.__init__(self, fname, "bias", p1, p2,
                             scale_infeatures=scale_infeatures)
        self.pmin = pmin
        self.pmax = pmax
        
    def __call__(self, layer):
        data = getattr(layer, self.attribute).data
        
        fanin = layer.in_features
        fanout = layer.out_features
        
        if self.fname == "normal":
            data.normal_(mean=self.p1, std=self.p2)            
        if self.fname == "uniform":
            data.uniform_(a=self.p1, b=self.p2)

        with torch.no_grad():
            data[0] = self.pmin
            data[-1] = self.pmax

    def update(self, transformer):
        """
        Update this object using the given pmin and pmax. They will be set to the
        attributes p1 and p2 respectively.        
        """
        
        # New data distribution is mean = 0, std = 1!!
        if transformer.scaling != '':
            self.p1 = 0
            self.p2 = 1  
        
            self.pmin = transformer.transform(
                self.pmin * np.ones((1, transformer.n_features_))).min()
            self.pmax = transformer.transform(
                self.pmax * np.ones((1, transformer.n_features_))).max()        

    def __str__(self):        
        s = "BiasInitializer("
        if self.fname == "normal":
            fnamestr = "N("
        if self.fname == "uniform":
            fnamestr = "U("
        s += f"{fnamestr}{self.p1}, {self.p2}), ({self.pmin}, {self.pmax}))"
        return s    

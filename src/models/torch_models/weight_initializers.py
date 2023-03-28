import torch, numpy as np

def initializers_from_string(s):
    """
    Receives either:
    s = '[]'
    s = ['Initializer(N(mu, sigma))']
    s = ['BiasInitializer(N(mu, sigma), pmin, pmax)']
    s = ['Initializer(N(mu, sigma))', 'BiasInitializer(N(mu, sigma), pmin, pmax)']

    Parse and return the object(s)
    """
    if s == '[]':
        return []

    s = s[1:-1]
    if len(s.split('Initializer')) == 2:
        if len(s.split('BiasInitializer')) == 2:
            return [BiasInitializer_from_string(s), ]
        else:
            return [Initializer_from_string(s), ]
    else:
        _, init, bias = s.split('Initializer')
        init_s = 'Initializer' + init.split("', 'Bias")[0]
        bias_s = 'BiasInitializer' + bias[:-1]
        return [Initializer_from_string(init_s),
                BiasInitializer_from_string(bias_s)]

def Initializer_from_string(s):
    """
    Receives  s = 'Initializer(N(mu, sigma))'
    parse the string and returns an Initializer object.
    """
    data = s[:-1].split('Initializer(')[1]
    
    distribution = data[0]
    data = data[2:-1]
    p1, p2 = data.split(", ")

    if distribution=="N":
        dstring = "normal"
    if distribution=="U":
        dstring = "uniform"

    return Initializer(dstring, "weight", p1, p2)    

def BiasInitializer_from_string(s):
    """
    Receives  s = 'BiasInitializer(N(mu, sigma), pmin, pmax)'
    parse the string and returns a BiasInitializer object.
    """
    data = s[:-1].split('BiasInitializer(')[1]
    
    distribution = data[0]
    data = data[2:]
    p1, p2, pmin, pmax = data.split(", ")
    p2 = p2[:-1]
    
    if distribution=="N":
        dstring = "normal"
    if distribution=="U":
        dstring = "uniform"

    return BiasInitializer(dstring, p1, p2, pmin, pmax)        
    
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

        with torch.no_grad():
            if self.fname == "normal":
                data.normal_(mean=self.p1, std=self.p2)            
            if self.fname == "uniform":
                data.uniform_(a=self.p1, b=self.p2)

        print(f"Initializing {layer} {self.attribute} using {self._str()}")

    def update(self, transformer):
        """
        Transform attributes of this object using the given transformer.
        """
        if transformer.scaling != '':
            print(f"Disabling Weight initialization since transformer={transfomer.scaling}")
            self.p1 = 0
            self.p2 = 1

    def _str(self, round_=False):
        s = "Initializer("
        if self.fname == "normal":
            fnamestr = "N("
        if self.fname == "uniform":
            fnamestr = "U("
        p1s = self.p1
        p2s = self.p2        
        if round_:
            p1s = round(p1s, ndigits=2)
            p2s = round(p2s, ndigits=2)
            
        s += f"{fnamestr}{p1s}, {p2s}))"
        return s

    def a_copy(self):
        return Initializer(self.fname, self.attribute,
                           self.p1, self.p2, scale_infeatures=self.scale_infeatures)

    def __repr__(self):
        return self._str(round_=True)
    
    def __str__(self):
        return self._str(round_=False)

        
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
        
        with torch.no_grad():
            if self.fname == "normal":
                data.normal_(mean=self.p1, std=self.p2)            
            if self.fname == "uniform":
                data.uniform_(a=self.p1, b=self.p2)
            
            data[0] = self.pmin
            data[-1] = self.pmax
 
        print(f"Initializing {layer} {self.attribute} using {self._str()}")

    def update(self, transformer):
        """
        Update this object using the given pmin and pmax. They will be set to the
        attributes p1 and p2 respectively.        
        """        
        # New data distribution is mean = 0, std = 1!!
        if transformer.scaling != '':
            print(f"Disabling Weight initialization since transformer={transformer.scaling}")
            
            self.p1 = 0
            self.p2 = 1  
        
            self.pmin = transformer.transform(
                self.pmin * np.ones((1, transformer.n_features_))).min()
            self.pmax = transformer.transform(
                self.pmax * np.ones((1, transformer.n_features_))).max()

    def a_copy(self):
        return BiasInitializer(self.fname,
                               self.p1, self.p2, self.pmin, self.pmax,
                               scale_infeatures=self.scale_infeatures)

    def _str(self, round_=False):
        s = "BiasInitializer("
        if self.fname == "normal":
            fnamestr = "N("
        if self.fname == "uniform":
            fnamestr = "U("
        p1s = self.p1
        p2s = self.p2
        pmins = self.pmin
        pmaxs = self.pmax
        if round_:
            p1s = round(p1s, ndigits=2)
            p2s = round(p2s, ndigits=2)
            pmins = round(pmins, ndigits=2)
            pmaxs = round(pmaxs, ndigits=2)            

        s += f"{fnamestr}{self.p1}, {self.p2}), {self.pmin}, {self.pmax})"
        return s

    def __repr__(self):
        return self._str(round_=True)    
            
    def __str__(self):        
        return self._str(round_=False)

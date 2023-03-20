import scipy.stats as stats, copy
from scipy.stats import rv_continuous
from src.models.samplers.structure_sampler import structure_sampler
from src.models.samplers.cnn_structure_sampler import flatten_dict

from src.models.torch_models.weight_initializers import Initializer, BiasInitializer

class wibi_sampler(rv_continuous):
    """
    Samples a weight + a Bias initializer
    """
    def __init__(self, pmin, pmax, wi_mean_sampling=(0, 1), wi_std_sampling=(3, 1),
                 wi_a_sampling=(-20, 0), wi_b_sampling=(0, 20),
                 bi_mean_sampling=(30, 5), bi_std_sampling=(40, 5),
                 bi_a_sampling=-20, bi_b_sampling=100):
        rv_continuous.__init__(self)

        self.pmin = pmin
        self.pmax = pmax
        
        self.wi_sampler = wi_sampler(
            mean_sampling=wi_mean_sampling, std_sampling=wi_std_sampling,
            a_sampling=wi_a_sampling, b_sampling=wi_b_sampling)
        self.bi_sampler = bi_sampler(pmin, pmax,
            mean_sampling=bi_mean_sampling, std_sampling=bi_std_sampling,
            a_sampling=bi_a_sampling, b_sampling=bi_b_sampling)

    def rvs(self, size=1, random_state=None):
        res = []
        for i in range(size):
            wi = self.wi_sampler.rvs(size=1, random_state=None)[0]
            bi = self.bi_sampler.rvs(size=1, random_state=None)[0]

            res.append([wi, bi])
            
        if size == 1:
            return res[0]
        return res

class wi_sampler(rv_continuous):
    """
    Samples a Weight intializer object. 
    First choose between normal and uniform distribution for weight initializing, 
    Then samples p1 and p2 accordingly (as mean and std for normal law, a and b
    if uniform)
    """
    def __init__(self, mean_sampling=(0, 1), std_sampling=(3, 1),
                 a_sampling=(-20, 0), b_sampling=(0, 20)):
        rv_continuous.__init__(self)
        
        self.distribution_sampler = stats.bernoulli(0.5)
        self.mean_sampler = stats.norm(mean_sampling[0], mean_sampling[1])
        self.std_sampler = stats.norm(std_sampling[0], std_sampling[1])
        self.a_sampler = stats.uniform(a_sampling[0], a_sampling[1] - a_sampling[0])
        self.b_sampler = stats.uniform(b_sampling[0], b_sampling[1] - b_sampling[0])
        
        self.choices = ["normal", "uniform"]

    def rvs(self, size=1, random_state=None):
        sampled = []
        for i in range(size):
            fname = self.choices[self.distribution_sampler.rvs(
                size=1,random_state=random_state)[0]]
            
            if fname == "normal":
                p1 = self.mean_sampler.rvs(size=1, random_state=random_state)[0]
                p2 = self.std_sampler.rvs(size=1, random_state=random_state)[0]
            if fname == "uniform":
                p1 = self.a_sampler.rvs(size=1, random_state=random_state)[0]
                p2 = self.b_sampler.rvs(size=1, random_state=random_state)[0]

            sampled.append(Initializer(fname, "weight", p1, p2))
            
        return sampled

    
class bi_sampler(rv_continuous):
    """
    Samples a Bias intializer object. 
    First choose between normal and uniform distribution for weight initializing, 
    Then samples p1 and p2 accordingly (as mean and std for normal law, a and b
    if uniform)
    """
    def __init__(self, pmin, pmax, mean_sampling=(30, 5), std_sampling=(40, 5),
                 a_sampling=-20, b_sampling=100):
        rv_continuous.__init__(self)
        
        self.pmin = pmin
        self.pmax = pmax
        
        self.distribution_sampler = stats.bernoulli(0.5)
        self.mean_sampler = stats.norm(mean_sampling[0], mean_sampling[1])
        self.std_sampler = stats.norm(std_sampling[0], std_sampling[1])
        self.a_sampler = stats.uniform(pmin, a_sampling - pmin)
        self.b_sampler = stats.uniform(b_sampling, pmax - b_sampling)
        
        self.choices = ["normal", "uniform"]

    def rvs(self, size=1, random_state=None):
        sampled = []
        for i in range(size):
            fname = self.choices[self.distribution_sampler.rvs(
                size=1,random_state=random_state)[0]]
            
            if fname == "normal":
                p1 = self.mean_sampler.rvs(size=1, random_state=random_state)[0]
                p2 = self.std_sampler.rvs(size=1, random_state=random_state)[0]
            if fname == "uniform":
                p1 = self.a_sampler.rvs(size=1, random_state=random_state)[0]
                p2 = self.b_sampler.rvs(size=1, random_state=random_state)[0]

            sampled.append(BiasInitializer(
                fname, p1, p2, self.pmin, self.pmax))

        return sampled    
            
                
        

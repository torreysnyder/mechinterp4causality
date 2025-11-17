
from pyro.distributions import Bernoulli, Uniform
import torch
import pyro
from pyro.ops.indexing import Vindex
from pyro.infer import config_enumerate

import hydra
from omegaconf import DictConfig

from causal_transformers.paths import CONFIG_PATH
from causal_transformers.utils import set_random_seed, load_worlds


"""
all the stochasticity is in the noise terms u that are "outside" of the model

this is useful for counterfactual inference where:
1) we find posterior u values (conditioning)
2) apply intervention ("doing")
3) simulate the model with the conditioned u values

"""

PROBS = torch.tensor([1e-3, 1-1e-3])

# this is the pyro (causal) model
@config_enumerate
def selection_causal_model(world_params):

    connectivity = world_params["connectivity"]
    cond_probs = world_params["cond_probs"]
    n_nodes = world_params["n_nodes"]

    values = {}

    # we can sample in the order [0,...,n_nodes-1] because the connectivity matrix is upper triangular
    for destination in range(n_nodes):

        # getting conditonal probability based on the source values
        sources = torch.where(connectivity[:, destination] == 1)[0].tolist()
        source_values = [values[str(source)].int() for source in sources]
        index = tuple([...] + source_values)
        p = Vindex(cond_probs[destination])[index]

        # sampling the destination variable
        name = str(destination)
        u = pyro.sample(name + "_u", Uniform(0,1)) # sampling the exogenous noise term U for this node
        value = u < p # endogenous variable is a deterministic function of U and the source values

        # to condition on observations, we need to use pyro.sample, so creating a very sharp bernoulli
        prob = Vindex(PROBS)[value.long()]
        # prob = 1e-3 if value == 0 else 1 - 1e-3
        values[name] = pyro.sample(name, Bernoulli(prob))

    return values



@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def test_causal_model(cfg : DictConfig):
    set_random_seed(cfg.seed)

    worlds = load_worlds()
    print(worlds)
    world_params = worlds["W099"]

    values = causal_model(world_params)
    print(values)

    predictive = pyro.infer.Predictive(causal_model, num_samples=1000)
    samples = predictive(world_params)
    for key in samples.keys():
        print(key, samples[key].mean())


if __name__ == "__main__":
    test_causal_model()

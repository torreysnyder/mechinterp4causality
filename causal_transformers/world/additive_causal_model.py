
from causal_transformers.world.additive import sample_additive_world
from pyro.distributions import Bernoulli, Uniform
import torch
import pyro
from pyro.ops.indexing import Vindex
from pyro.infer import config_enumerate

import hydra
from omegaconf import DictConfig

from causal_transformers.paths import CONFIG_PATH
from causal_transformers.utils.dataset_utils import set_random_seed, load_worlds


PROBS = torch.tensor([1e-4, 1-1e-4])

# this is the pyro (causal) model
@config_enumerate
def additive_causal_model(world_params):

    # connectivity = world_params["connectivity"]
    weights = world_params["weights"].float() # edge weights that are summed to get the child's probability
    # cond_probs = world_params["cond_probs"]
    n_nodes = world_params["n_nodes"]

    values = {}

    # we can sample in the order [0,...,n_nodes-1] because the connectivity matrix is upper triangular
    for destination in range(n_nodes):

        # calculating the destination probability based on the source values
        score = 0.0 # the score is passed through a sigmoid to get the probability
        score += weights[destination][destination] # bias term on the diagonal for the destination (independent of everything else)
        score += torch.dot(weights[:destination, destination], torch.tensor([values[str(source)] for source in range(destination)]))

        p = torch.sigmoid(score)

        # sampling the destination variable
        name = str(destination)
        u = pyro.sample(name + "_u", Uniform(0,1)) # sampling the exogenous noise term U for this node
        value = u < p # endogenous variable is a deterministic function of U and the source values

        # to condition on observations, we need to use pyro.sample, so creating a very sharp bernoulli
        prob = Vindex(PROBS)[value.long()]
        values[name] = pyro.sample(name, Bernoulli(prob))

    return values



@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def test_causal_model(cfg : DictConfig):
    set_random_seed(cfg.seed)

    world_params = sample_additive_world(cfg)
    print(world_params["weights"])
    world_params["weights"][0,0] = -10.0

    values = additive_causal_model(world_params)
    print(values)

    predictive = pyro.infer.Predictive(additive_causal_model, num_samples=1000)
    samples = predictive(world_params)
    for key in samples.keys():
        print(key, samples[key].mean())


if __name__ == "__main__":
    test_causal_model()


import torch

import hydra
from omegaconf import DictConfig

from itertools import product
import numpy as np

from causal_transformers.paths import CONFIG_PATH, ROOT_DIR
from causal_transformers.utils import set_random_seed

import matplotlib.pyplot as plt

"""

world are DAGs (Bernoulli random variables), for each world:
- sample the number of nodes
- sample connectivity (0 or normally distributed value)

additive worlds consist of a set of nodes taking binary values with sparse edges that are normally distributed
children are true with probability sigmoid(sum(edges to parents) + bias(child))
"""
def sample_additive_world(cfg):
    n_nodes = torch.randint(cfg.dataset.world.n_nodes.min, cfg.dataset.world.n_nodes.max+1, (1,)).item()

    # sample upper triangular weight matrix (directed acyclic graph)
    # rows are sources
    # columns are targets

    # we sample weights uniformly
    possible_weights = torch.tensor(cfg.dataset.world.weights)
    weights = torch.tensor(np.random.choice(possible_weights, size=(n_nodes, n_nodes)))
    weights = torch.triu(weights, diagonal=0)

    # connectivity is a binary mask represeting if there is a connection between nodes
    connectivity = torch.triu(weights != 0.0, diagonal=1) # excluding the diagonal (v7 dataset onwards, v6 still has diagonal)

    # save the world (SCM) params
    world_params = {
        "n_nodes": n_nodes,
        # "connections_per_node": connections_per_node,
        # "connection_prob": connection_prob,
        "connectivity": connectivity,
        "weights": weights,
    }

    return world_params




@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def render_sampled_causal_model(cfg):
    from causal_transformers.utils import render_world, set_random_seed, render_causal_model
    # from causal_transformers.world.additive.additive_causal_model import additive_causal_model
    import matplotlib.pyplot as plt

    set_random_seed(cfg.seed)

    # cfg.world = cfg.dataset.world
    dir = ROOT_DIR / "data" / "figures" / "additive_world"
    dir.mkdir(parents=True, exist_ok=True)

    world_params = sample_additive_world(cfg)

    # render connectivity matrix
    plt.imshow(world_params["connectivity"])
    plt.savefig(dir / "connectivity.pdf")

    plt.imshow(world_params["weights"])
    plt.savefig(dir / "weights.pdf")
    plt.close()

    # render causal graph using graphviz
    render_world(cfg, world_params, dir / "additive_world.pdf", render_weights=True)

    # pyro model with noise terms
    # render_causal_model(additive_causal_model, (world_params,), filename= dir /  "pyro_additive_causal_model.pdf")

if __name__ == "__main__":
    render_sampled_causal_model()

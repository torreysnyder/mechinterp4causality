import torch
import numpy as np
from causal_transformers.utils.dataset_utils import load_worlds


def get_world_connectivities(cfg):
    worlds = load_worlds(cfg)
    world_connectivities = torch.zeros((cfg.world.n_worlds, cfg.dataset.world.n_nodes.max, cfg.dataset.world.n_nodes.max))
    for world_index in range(cfg.world.n_worlds):
        connectivity = worlds[world_index]["connectivity"]
        n_nodes = connectivity.shape[0]
        world_connectivities[world_index, :n_nodes, :n_nodes] = connectivity
    return world_connectivities


def get_max_n_sources(cfg):
    worlds = load_worlds(cfg)
    max_n_sources = 0
    for world_index in worlds:
        world_params = worlds[world_index]
        connectivity = world_params["connectivity"]
        n_sources = torch.sum(connectivity, dim=0)
        max_n_sources = max(max_n_sources, torch.max(n_sources).item())
    return int(max_n_sources)


def get_world_cond_probs(cfg):
    worlds = load_worlds(cfg)
    max_n_sources = get_max_n_sources(cfg)
    world_cond_probs = torch.zeros((cfg.world.n_worlds, cfg.dataset.world.n_nodes.max, 2**max_n_sources))
    for world_index in worlds:
        world_params = worlds[world_index]
        cond_probs = world_params["cond_probs"]
        for node in cond_probs:
            ps = cond_probs[node]
            ps_flat = ps.flatten()
            world_cond_probs[world_index, node, :len(ps_flat)] = ps_flat
    return world_cond_probs


def get_causal_model(cfg):
    from causal_transformers.world import additive_causal_model, selection_causal_model
    return additive_causal_model if cfg.dataset.type == "additive" else selection_causal_model


def get_prob_indices(cfg):
    from causal_transformers.dataset.preprocessor import Preprocessor
    preprocessor = Preprocessor(cfg)
    return torch.tensor([preprocessor.vocab[f"{prob:.2f}"] for prob in np.arange(0.0, 1.0, 0.01)])

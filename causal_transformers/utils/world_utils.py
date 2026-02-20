import torch
import numpy as np


def get_causal_model(cfg):
    from causal_transformers.world import additive_causal_model, selection_causal_model
    return additive_causal_model if cfg.dataset.type == "additive" else selection_causal_model


def get_prob_indices(cfg):
    from causal_transformers.dataset.preprocessor import Preprocessor
    preprocessor = Preprocessor(cfg)
    return torch.tensor([preprocessor.vocab[f"{prob:.2f}"] for prob in np.arange(0.0, 1.0, 0.01)])

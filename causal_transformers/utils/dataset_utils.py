import torch
from torch.utils.data import DataLoader
import numpy as np
import string
from pathlib import Path


def load_held_out_world_indices(cfg):
    """Load indices specifying which worlds/SCMs are held out (not used for training)
    returns tensor of indices
    """
    return torch.load("causal_transformers/dataset/data/datasets/gaussian_dataset_v6/worlds/held_out_world_indices.pt", weights_only=False)


def custom_collate_fn(batch):
    """
    custom collation for DataLoader
    each batch item is assumed to be:
        (indices, target_probs, meta_data)
    where indices and target_probs are themselves iterable collections
    flatten indices/target_probs across items while keeping metadata per-item
    :param batch:
    :return:
    """
    indices_list = []
    target_probs_list = []
    meta_data_list = []
    for indices, target_probs, meta_data in batch:
        # indices: list/tuple of tensors; flatten into one big list
        indices_list.extend(indices)
        # target_probs: list/tuple of tensors; flatten into one big list
        target_probs_list.extend(target_probs)
        # keep metadata aligned with original items
        meta_data_list.append(meta_data)
    # stack all indices into one tensor: [N_total, ...]
    indices = torch.stack(indices_list)
    # concatenate all target probs likewise
    target_probs = torch.cat(target_probs_list)
    return indices, target_probs, meta_data_list


def get_dataloaders(cfg, batch_size, shuffle_all=False, subset_size=None, example_type=None):
    """
    Build DataLoaders for train/valid/test splits
    only gaussian datasets are currently supported
    Args:
    :param cfg: hydra/omegaconf config with dataset + training settings
    :param batch_size: batch size for each loader
    :param shuffle_all: if True, also shuffle valid/test (normally only train)
    :param subset_size: if set, dataset loads only this many examples
    :param example_type: optional filter (passed to dataset constructor)
    :return:
    """
    from causal_transformers.dataset.causal_dataset_gaussian import CausalDatasetGaussian
    splits = ["train", "valid", "test"]
    dataloaders = {}
    for split in splits:
        if cfg.dataset.type == "gaussian":
            dataset = CausalDatasetGaussian(cfg, split=split, subset_size=subset_size, example_type=example_type)
        else:
            raise ValueError(f"dataset type {cfg.dataset.type} not recognized")
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, num_workers=cfg.train.num_workers, shuffle=True if split == "train" or shuffle_all else False)
    return dataloaders


def get_d_vocab(cfg):
    """
    Compute vocab size for tokenizer used by model
    Includes:
    - base special tokens
    - value-encoding bins
    - infinities (+INF/-INF)
    - node-index tokens
    - uppercase variable-name tokens
    """
    n_tokens = 8 # base special tokens
    if cfg.dataset.type == "additive":
        n_tokens += 101 # additive uses fixed extra tokens
    elif cfg.dataset.type == "gaussian":
        # discretized numeric values
        n_tokens += cfg.dataset.value_encoding.steps
        # sentinel tokens for out-of-range values
        n_tokens += 2
    else:
        raise ValueError(f"dataset type {cfg.dataset.type} not recognized")
    # tokens for node IDs
    n_tokens += cfg.dataset.world.n_nodes.max
    # tokens for variable names A-z
    n_tokens += len(string.ascii_uppercase)
    return n_tokens


def get_n_ctx(cfg):
    """
    Compute maximum context length a sequence might need
    This is based on config bounds on:
        - SCM code width
        - number of data examples and interventions
        - number of inference observations/interventions
        - whether examples are single-variable or multi-variable

    """
    # start token
    n_ctx = 1
    # fixed-length SCM code prefix
    n_ctx += cfg.dataset.names.scm_code_width
    assert cfg.dataset.type == "gaussian", "need to reimplement the additive get_n_ctx function"
    # ---- Data-example context length estimate ----
    # each intervention contributes ~3 tokens
    n_ctx_data = max(cfg.dataset.data_examples.n_interventions) * 3
    # eachh data example length depends on whether values were sampled
    n_len_data_example = 2 if cfg.dataset.data_examples.sample else 3
    n_data_nodes = 1 if cfg.dataset.data_examples.single_variable else cfg.dataset.world.n_nodes.max
    n_ctx_data += n_len_data_example * n_data_nodes
    n_ctx_data += 1
    n_ctx_inference = (max(cfg.dataset.inference_examples.n_observations) + max(cfg.dataset.inference_examples.n_interventions)) * 3
    n_len_inference_example = 3
    n_inference_nodes = 1 if cfg.dataset.inference_examples.single_variable else cfg.dataset.world.n_nodes.max
    n_ctx_inference += n_len_inference_example * n_inference_nodes
    n_ctx += max(n_ctx_data, n_ctx_inference)
    return n_ctx


def get_world_weights(cfg):
    world_weights_path = Path("causal_transformers/dataset/data/datasets/gaussian_dataset_v6/worlds/world_weights.pt")

    if world_weights_path.exists():
        world_weights = torch.load(world_weights_path, weights_only=False)
    else:
        raise Exception(f"world weights file {world_weights_path} does not exist")
    return world_weights


def encode_value(cfg, value, preprocessor):
    if torch.is_tensor(value):
        value = value.numpy()
    if value < cfg.dataset.value_encoding.range[0]:
        return "-INF"
    elif value > cfg.dataset.value_encoding.range[1]:
        return "+INF"
    else:
        value = preprocessor.values[np.abs(preprocessor.values - np.array(value)).argmin()]
        value = 0.0 if abs(value) == 0 else value
        decimal_points = cfg.dataset.value_encoding.decimal_points
        return f"{value:.{decimal_points}f}"


def sample_from_multivariate_normal(means, cov,
                                    eps: float = 1e-8,
                                    significant_thresh: float = 1e-5):
    """
    Draw a sample from a (possibly slightly non-PSD) covariance matrix.

    - Symmetrises the covariance.
    - If the minimum eigenvalue is slightly negative (>= -significant_thresh),
      treat it as numerical noise and fix with jitter.
    - If it's more negative than that, raise an error.
    """
    # Make sure the covariance is exactly symmetric
    cov_symmetric = (cov + cov.T) / 2

    eigvals = np.linalg.eigvals(cov_symmetric)
    min_eig = np.min(eigvals)

    # Truly bad covariance: large negative eigenvalues
    if min_eig < -significant_thresh:
        raise ValueError(
            f"Covariance matrix has significant negative eigenvalues: {min_eig}"
        )

    # Small negative / near-zero eigenvalues: fix with jitter
    if min_eig <= eps:
        # Enough jitter to push the smallest eigenvalue above eps
        jitter = max(eps, -min_eig + eps)
        cov_symmetric = cov_symmetric + np.eye(cov_symmetric.shape[0]) * jitter

    try:
        samples = np.random.multivariate_normal(means, cov_symmetric)
    except np.linalg.LinAlgError:
        # Fallback: progressively increase jitter if the decomposition still fails
        jitter = 1e-6
        while jitter < 1e-2:
            try:
                cov_fixed = cov_symmetric + np.eye(cov_symmetric.shape[0]) * jitter
                samples = np.random.multivariate_normal(means, cov_fixed)
                print(
                    f"Warning: Added jitter of {jitter} to covariance matrix "
                    f"to ensure positive definiteness"
                )
                break
            except np.linalg.LinAlgError:
                jitter *= 10
        else:
            raise ValueError(
                "Failed to create a usable covariance matrix even with jitter"
            )

    return samples



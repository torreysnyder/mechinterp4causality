import torch


from causal_transformers.utils import ROOT_DIR, DATASETS_DIR
from torch.utils.data import DataLoader
import numpy as np
import string


def load_worlds(cfg):
    raise Exception("This function is not used anymore. Use get_world_weights instead.")
    print("loading worlds (SCMs)...")
    worlds = torch.load(DATASETS_DIR / cfg.dataset.name / "worlds" / "worlds.pt", weights_only=True)
    print("done laoding worlds")
    return worlds


def load_held_out_world_indices(cfg):
    return torch.load(DATASETS_DIR / cfg.dataset.name / "worlds" / "held_out_world_indices.pt", weights_only=True)


def load_dataset_torch(cfg):
    dataset_dir = DATASETS_DIR / cfg.dataset.name
    return torch.load(dataset_dir / "dataset.pt")


def custom_collate_fn(batch):
    indices_list = []
    target_probs_list = []
    meta_data_list = []
    for indices, target_probs, meta_data in batch:
        indices_list.extend(indices)
        target_probs_list.extend(target_probs)
        meta_data_list.append(meta_data)
    indices = torch.stack(indices_list)
    target_probs = torch.cat(target_probs_list)
    return indices, target_probs, meta_data_list


def get_dataloaders(cfg, batch_size, shuffle_all=False, subset_size=None, example_type=None):
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
    n_tokens = 8
    if cfg.dataset.type == "additive":
        n_tokens += 101
    elif cfg.dataset.type == "gaussian":
        n_tokens += cfg.dataset.value_encoding.steps
        n_tokens += 2
    else:
        raise ValueError(f"dataset type {cfg.dataset.type} not recognized")
    n_tokens += cfg.dataset.world.n_nodes.max
    n_tokens += len(string.ascii_uppercase)
    return n_tokens


def get_n_ctx(cfg):
    n_ctx = 1
    n_ctx += cfg.dataset.names.scm_code_width
    assert cfg.dataset.type == "gaussian", "need to reimplement the additive get_n_ctx function"
    n_ctx_data = max(cfg.dataset.data_examples.n_interventions) * 3
    n_len_data_example = 2 if cfg.dataset.data.examples.sample else 3
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
    world_weights_path = DATASETS_DIR/cfg.dataset.name/"worlds"/"world_weights.pt"
    if world_weights_path.exists():
        world_weights = torch.load(world_weights_path, weights_only=True)
    else:
        raise Exception(f"world weights file {world_weights_path} does not exist")
    return world_weights


def encode_value(cfg, value, preprocesor):
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


def sample_from_multivariate_normal(means, cov):
    cov_symmetric = (cov + cov.T)/2
    eigvals = np.linalg.eigvals(cov_symmetric)
    if np.any(eigvals < -1e-8):
        raise ValueError(f"Covariance matrix has significant negative eigenvalues: {eigvals.min()}")
    if np.any(eigvals <= 1e-8):
        cov_symmetric += np.eye(cov_symmetric.shape[0]) * max(1e-8, abs(eigvals.min()) + 1e-8)
    try:
        samples = np.random.multivariate_normal(means, cov_symmetric)
    except np.linalg.LinAlgError:
        jitter = 1e-6
        while jitter < 1e-2:
            try:
                cov_fixed = cov_symmetric + np.eye(cov_symmetric.shape[0]) * jitter
                samples = np.random.multivariate_normal(means, cov_fixed)
                print(f"Warning: Added jitter of {jitter} to covariance matrix to ensure positive definiteness")
                break
            except np.linalg.LinAlgError:
                jitter *= 10
        else:
            raise ValueError("Failed to create a usable covariance matrix even with jitter")
    return samples


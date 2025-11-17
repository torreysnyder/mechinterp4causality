import torch
from torch.utils.data import Dataset
import causal_transformers as ct
import numpy as np
import math
import hydra

class CausalDatasetGaussian(Dataset):
    def __init__(self, cfg, split="train", subset_size=None, example_type=None):
        self.cfg = cfg
        self.split = split
        self.preprocessor = ct.dataset.Preprocessor(cfg)
        self.vocab_size = len(self.preprocessor.vocab)
        self.world_weights = ct.utils.get_world_weights(cfg)
        n_worlds = self.world_weights.shape[0]
        self.held_out_world_indices = ct.utils.load_held_out_world_indices(cfg).numpy()
        if cfg.dataset.dynamic:
            all_indices = np.arange(len(self.world_weights))
            if self.split == "train":
                self.data_indices = all_indices
                mask = np.ones(n_worlds, dtype=bool)
                mask[self.held_out_world_indices] = False
                self.inference_indices = np.arange(n_worlds)[mask]
            elif self.split == "valid":
                self.data_indices = np.random.permutation(all_indices)[:cfg.dataset.data_examples.n_valid_worlds]
                self.inference_indices = np.random.permutation(all_indices)[:cfg.dataset.inference_examples.n_valid_worlds]
            elif self.split == "test":
                self.data_indices = self.held_out_world_indices.copy()
                self.inference_indices = self.held_out_world_indices.copy()
            self.n_data = len(self.data_indices) * self.cfg.dataset.data_examples.n_reps
            self.n_inference = len(self.inference_indices) * self.cfg.dataset.inference_examples.n_reps
        else:
            raise NotImplementedError("non dynamic version is not used anymore")
        if example_type is not None:
            if example_type == "data":
                self.n_inference = 0
            elif example_type == "inference":
                self.n_data = 0
            else:
                raise ValueError(f"example_type {example_type} not recognized")
        self.len = self.n_data + self.n_inference
        if subset_size is not None:
            self.len = min(self.len, subset_size)

    def decode_indices(self, indices):
        return self.preprocessor.decode_indices(indices)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx < self.len, f"Index {idx} out of bounds for dataset of length {self.len}"
        example_type = "data" if idx < self.n_data else "inference"
        example_idx = idx if example_type == "data" else idx - self.n_data
        result = self.get_string(example_type, example_idx)
        indices = self.preprocessor(result["string"])
        n_observations = (~torch.isnan(result["observations"])).sum()
        n_interventions = (~torch.isnan(result["interventions"])).sum()
        meta_data = {
            "dataset_id": idx,
            "example_type": example_type,
            **result,
            "n_observations": n_observations,
            "n_interventions": n_interventions,
        }
        return indices, meta_data

    def get_dynamic_result(self, scm_index, string_type):
        if string_type == "data":
            n_observations = 0
            n_interventions = np.random.choice(self.cfg.dataset.data_examples.n_interventions)
        else:
            n_observations = np.random.choice(self.cfg.dataset.inference_examples.n_observations)
            n_interventions = np.random.choice(self.cfg.dataset.inference_examples.n_interventions)
        n_nodes = self.cfg.dataset.world.n_nodes.max
        observation_interval = self.cfg.dataset.observation_interval
        intervention_interval = self.cfg.dataset.intervention_interval
        observation = torch.full((n_nodes,), float('nan'))
        intervention = torch.full((n_nodes,), float('nan'))
        if n_observations > 0:
            observed_nodes = np.random.permutation(n_nodes)[:n_observations]
            observed_values = torch.rand((n_observations,))
            observed_values = observed_values * (observation_interval[1] - observation_interval[0]) + observation_interval[0]
            for (node, value) in zip(observed_nodes, observed_values):
                observation[node] = value
        if n_interventions > 0:
            intervened_nodes = np.random.permutation(n_nodes)[:n_interventions]
            intervened_values = torch.rand((n_interventions,))
            intervened_values = intervened_values * (intervention_interval[1] - intervention_interval[0]) + intervention_interval[0]
            for (node, value) in zip(intervened_nodes, intervened_values):
                intervention[node] = value
        weights = self.world_weights[scm_index]
        result = ct.inference.gaussian_scm_inference(weights, observation, intervention, self.cfg.dataset.world.noise_var)
        return observation.numpy(), intervention.numpy(), result["means"].numpy(), result["cov"].numpy()

    def get_string(self, string_type, idx):
        s = "DATA" if string_type == "data" else "INFERENCE"
        if self.cfg.dataset.dynamic:
            if string_type == "data":
                scm_index = self.data_indices[idx // self.cfg.dataset.data_examples.n_reps]
            elif string_type == "inference":
                scm_index = self.inference_indices[idx // self.cfg.dataset.inference_examples.n_reps]
            observations, interventions, means, cov = self.get_dynamic_result(scm_index, string_type)
        else:
            raise NotImplementedError("non dynamic version is not used anymore")
        scm_letters = ct.utils.encode_scm_index(self.cfg, scm_index)
        for letter in scm_letters:
            s += f" {letter}"
        interventions_observations = []
        for (node_index, value) in enumerate(interventions):
            if not math.isnan(value):
                node_name = ct.utils.get_remapped_node_name(self.cfg, scm_index, node_index)
                interventions_observations.append(f" DO {node_name} {ct.utils.encode_value(self.cfg, value, self.preprocessor)}")
        for (node_index, value) in enumerate(observations):
            if not math.isnan(value):
                node_name = ct.utils.get_remapped_node_name(self.cfg, scm_index, node_index)
                interventions_observations.append(f" OBS {node_name} {ct.utils.encode_value(self.cfg, value, self.preprocessor)}")
        for i in np.random.permutation(len(interventions_observations)):
            s += interventions_observations[i]
        n_nodes = len(means)
        if self.cfg.dataset.data_examples.permute_variable_order:
            node_order = np.random.permutation(n_nodes)
        else:
            node_order = range(n_nodes)
        if string_type == "data":
            if self.cfg.dataset.data_examples.sample:
                samples = torch.tensor(ct.utils.sample_from_multivariate_normal(means, cov))
            else:
                samples = means
            for node_index in node_order:
                node_name = ct.utils.get_remapped_node_name(self.cfg, scm_index, node_index)
                node_value = ct.utils.encode_value(self.cfg, samples[node_index], self.preprocessor)
                s += f" {node_name}{node_value}"
        elif string_type == "inference":
            for node_index in node_order:
                node_name = ct.utils.get_remapped_node_name(self.cfg, scm_index, node_index)
                mean_value = ct.utils.encode_value(self.cfg, means[node_index], self.preprocessor)
                std_value = ct.utils.encode_value(self.cfg, math.sqrt(cov[node_index, node_index]), self.preprocessor)
                s += f" {node_name} {mean_value} {std_value}"
        s += " EOS"
        if string_type == "inference":
            samples = torch.full((n_nodes,), float('nan'))
        result = {
            "string": s,
            "scm_index": torch.tensor(scm_index, dtype=torch.int32),
            "observations": torch.tensor(observations),
            "interventions": torch.tensor(interventions),
            "samples": samples.clone().detach(),
            "means": torch.tensor(means),
            "cov": torch.tensor(cov),
            "node_order": torch.tensor(node_order, dtype=torch.int32),
        }
        return result


@hydra.main(version_base=None, config_name='config', config_path=ct.paths.CONFIG_PATH)
def test_causal_dataset_gaussian(cfg):
    dataset = CausalDatasetGaussian(cfg, split="train")
    print("n data examples: ", dataset.n_data)
    print("n inference examples: ", dataset.n_inference)
    held_out_world_indices = ct.utils.held_out_world_indices(cfg)
    print("held out world indices: ", held_out_world_indices[:10])
    idx = 120020
    indices, meta_data = dataset[idx]
    print(meta_data["string"])
    print(meta_data["scm_index"])
    print(dataset.world_weights[meta_data["scm_index"]])
    print()
    random_indices = np.random.permutation(len(dataset))
    for i in random_indices[:10]:
        indices, meta_data = dataset[i]
        print(meta_data["string"])


if __name__ == "__main__":
    test_causal_dataset_gaussian()



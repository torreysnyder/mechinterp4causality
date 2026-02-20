import torch
from torch.utils.data import Dataset
import causal_transformers as ct
from causal_transformers.dataset.preprocessor import Preprocessor
from causal_transformers.inference import gaussian_scm_inference
from causal_transformers.paths import CONFIG_PATH
print("CONFIG_PATH =", CONFIG_PATH)
import numpy as np
import math
import hydra


class CausalDatasetGaussian(Dataset):
    """
    PyTorch Dataset for training transformers on causal inference tasks with Gaussian SCMs
    Generates 2 types of examples:
    1. DATA examples: Sample observations from SCMs (for learning causal structure)
    2. INFERENCE examples: Query conditional distributions given observations/interventions
    Each example is a text string encoding:
    - SCM identifier
    - Observations (OBS node value)
    - Interventions (DO node value)
    - Target: either samples (DATA) or mean/std predictions (INFERENCE)
    """
    def __init__(self, cfg, split="train", subset_size=None, example_type=None):
        """
        Initialize Gaussian causal dataset
        Args:
        :param cfg: Hydra configuration object
        :param split: "train", "valid" or "test"
        :param subset_size: Optional limit on dataset size
        :param example_type: Optional filter for "data" or "inference" examples only
        """
        self.cfg = cfg
        self.split = split
        # Preprocessor handles tokenization of strings into indices
        self.preprocessor = Preprocessor(cfg)
        self.vocab_size = len(self.preprocessor.vocab)
        # Load all SCM weight matrices (one per world/causal graph)
        self.world_weights = ct.utils.dataset_utils.get_world_weights(cfg)
        n_worlds = self.world_weights.shape[0]
        # Load held-out world indices for test set evaluation
        self.held_out_world_indices = ct.utils.dataset_utils.load_held_out_world_indices(cfg).numpy()
        if cfg.dataset.dynamic:
            all_indices = np.arange(len(self.world_weights))
            if self.split == "train":
                # training uses all worlds except held-out ones
                self.data_indices = all_indices
                mask = np.ones(n_worlds, dtype=bool)
                mask[self.held_out_world_indices] = False
                self.inference_indices = np.arange(n_worlds)[mask]
            elif self.split == "valid":
                # validation randomly samples from all worlds
                self.data_indices = np.random.permutation(all_indices)[:cfg.dataset.data_examples.n_valid_worlds]
                self.inference_indices = np.random.permutation(all_indices)[:cfg.dataset.inference_examples.n_valid_worlds]
            elif self.split == "test":
                # test uses only held-out worlds (distribution shift)
                self.data_indices = self.held_out_world_indices.copy()
                self.inference_indices = self.held_out_world_indices.copy()
            # calculate total number of examples
            # n_reps: how many examples to generate per world
            self.n_data = len(self.data_indices) * self.cfg.dataset.data_examples.n_reps
            self.n_inference = len(self.inference_indices) * self.cfg.dataset.inference_examples.n_reps
        else:
            raise NotImplementedError("non dynamic version is not used anymore")
        # optionally filter to only one example type
        if example_type is not None:
            if example_type == "data":
                self.n_inference = 0
            elif example_type == "inference":
                self.n_data = 0
            else:
                raise ValueError(f"example_type {example_type} not recognized")
        # total dataset length
        self.len = self.n_data + self.n_inference
        if subset_size is not None:
            self.len = min(self.len, subset_size)

    def decode_indices(self, indices):
        """Convert token indices back to readable string"""
        return self.preprocessor.decode_indices(indices)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        Get a single example from dataset
        :param idx: index (tokenized string)
        :return:
        indices: tokenized string as tensor
        meta_data: Dict containing all information about the example
        """
        assert idx < self.len, f"Index {idx} out of bounds for dataset of length {self.len}"
        # Determine if this is a DATA or INFERENCE example
        # first n_data indices are DATA, remaining are INFERENCE
        example_type = "data" if idx < self.n_data else "inference"
        example_idx = idx if example_type == "data" else idx - self.n_data
        # Generate the string representation and metadata
        result = self.get_string(example_type, example_idx)
        # tokenize the string into indices
        indices = self.preprocessor(result["string"])
        # Count non-NaN observations and interventions
        n_observations = (~torch.isnan(result["observations"])).sum()
        n_interventions = (~torch.isnan(result["interventions"])).sum()
        # package metadata
        meta_data = {
            "dataset_id": idx,
            "example_type": example_type,
            **result,
            "n_observations": n_observations,
            "n_interventions": n_interventions,
        }
        return indices, meta_data

    def get_dynamic_result(self, scm_index, string_type):
        """
        Generate observations, interventions and resulting distribution for an SCM
        this randomly samples:
        - which nodes to observe/intervene on
        - values for those observations/interventions
        then uses Gaussian SCM inference to compute resulting conditional distribution
        Args:
        :param scm_index: which world/SCM to use
        :param string_type: "data" or "inference"
        :return:
        observation: array of observed values (NaN for unobserved nodes)
        intervention: array of intervention values (NaN for non-intervened nodes)
        means: posterior mean for each node
        cov: posterior covariance matrix
        """
        if string_type == "data":
            # DATA examples have interventions but no observations
            n_observations = 0
            n_interventions = np.random.choice(self.cfg.dataset.data_examples.n_interventions)
        else:
            # INFERENCE examples have both observations and interventions
            n_observations = np.random.choice(self.cfg.dataset.inference_examples.n_observations)
            n_interventions = np.random.choice(self.cfg.dataset.inference_examples.n_interventions)
        n_nodes = self.cfg.dataset.world.n_nodes.max
        observation_interval = self.cfg.dataset.observation_interval
        intervention_interval = self.cfg.dataset.intervention_interval
        # initialize with NaN (indicating no observation/intervention)
        observation = torch.full((n_nodes,), float('nan'))
        intervention = torch.full((n_nodes,), float('nan'))
        # randomly select nodes to observe and their values
        if n_observations > 0:
            observed_nodes = np.random.permutation(n_nodes)[:n_observations]
            observed_values = torch.rand((n_observations,))
            observed_values = observed_values * (observation_interval[1] - observation_interval[0]) + observation_interval[0]
            for (node, value) in zip(observed_nodes, observed_values):
                observation[node] = value
        # randomly select nodes to intervene on and their values
        if n_interventions > 0:
            intervened_nodes = np.random.permutation(n_nodes)[:n_interventions]
            intervened_values = torch.rand((n_interventions,))
            intervened_values = intervened_values * (intervention_interval[1] - intervention_interval[0]) + intervention_interval[0]
            for (node, value) in zip(intervened_nodes, intervened_values):
                intervention[node] = value
        # use the SCM weight matrix to compute posterior distribution
        weights = self.world_weights[scm_index]
        result = ct.inference.gaussian_scm_inference.gaussian_scm_inference(weights, observation, intervention, self.cfg.dataset.world.noise_var)
        return observation.numpy(), intervention.numpy(), result["means"].numpy(), result["cov"].numpy()

    def get_string(self, string_type, idx):
        """
        Generate the string representation of an example
        String format:
        DATA/INFERENCE [SCM_ID] [DO node val]*[OBS node val]*[node val]*EOS
        for DATA: node values are samples from distribution
        for INFERENCE: node values are mean and std of distribution
        Args:
        string_type: "data" or "inference"
        :param idx: index within example type
        :return: Dictionary containing string and all metadata
        """
        # start with example type tag
        s = "DATA" if string_type == "data" else "INFERENCE"
        if self.cfg.dataset.dynamic:
            # map idx to which SCM/world to use
            if string_type == "data":
                scm_index = self.data_indices[idx // self.cfg.dataset.data_examples.n_reps]
            elif string_type == "inference":
                scm_index = self.inference_indices[idx // self.cfg.dataset.inference_examples.n_reps]
            # generate observations, interventions and resulting distribution
            observations, interventions, means, cov = self.get_dynamic_result(scm_index, string_type)
        else:
            raise NotImplementedError("non dynamic version is not used anymore")
        # encode SCM index as letters (ex: A B C)
        scm_letters = ct.utils.name_utils.encode_scm_index(self.cfg, scm_index)
        for letter in scm_letters:
            s += f" {letter}"
        # build intervention and observation strings
        interventions_observations = []
        # add interventions: "DO node_name value"
        for (node_index, value) in enumerate(interventions):
            if not math.isnan(value):
                node_name = ct.utils.name_utils.get_remapped_node_name(self.cfg, scm_index, node_index)
                interventions_observations.append(f" DO {node_name} {ct.utils.dataset_utils.encode_value(self.cfg, value, self.preprocessor)}")
        # add observations: "OBS node_name value"
        for (node_index, value) in enumerate(observations):
            if not math.isnan(value):
                node_name = ct.utils.name_utils.get_remapped_node_name(self.cfg, scm_index, node_index)
                interventions_observations.append(f" OBS {node_name} {ct.utils.dataset_utils.encode_value(self.cfg, value, self.preprocessor)}")
        # randomly permute order of observations and interventions
        for i in np.random.permutation(len(interventions_observations)):
            s += interventions_observations[i]
        # add target values (either samples or mean/std)
        n_nodes = len(means)
        if self.cfg.dataset.data_examples.permute_variable_order:
            node_order = np.random.permutation(n_nodes)
        else:
            node_order = range(n_nodes)
        if string_type == "data":
            # DATA: sample from distribution or use means
            if self.cfg.dataset.data_examples.sample:
                samples = torch.tensor(ct.utils.dataset_utils.sample_from_multivariate_normal(means, cov))
            else:
                samples = means
            # format: node_name value
            for node_index in node_order:
                node_name = ct.utils.name_utils.get_remapped_node_name(self.cfg, scm_index, node_index)
                node_value = ct.utils.dataset_utils.encode_value(self.cfg, samples[node_index], self.preprocessor)
                # NOTE: add a space between node name and value
                s += f" {node_name} {node_value}"
        elif string_type == "inference":
            # INFERENCE: output mean and standard deviation
            # format: "node_name mean std"
            for node_index in node_order:
                node_name = ct.utils.name_utils.get_remapped_node_name(self.cfg, scm_index, node_index)
                mean_value = ct.utils.dataset_utils.encode_value(self.cfg, means[node_index], self.preprocessor)
                std_value = ct.utils.dataset_utils.encode_value(self.cfg, math.sqrt(cov[node_index, node_index]), self.preprocessor)
                s += f" {node_name} {mean_value} {std_value}"
        # end of sequence marker
        s += " EOS"
        # for inference examples samples are not available
        if string_type == "inference":
            samples = torch.full((n_nodes,), float('nan'))
        # package all metadata
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

class ActivationPatchingDataset:
    VARIABLES = ['V1', 'V2', 'V3', 'V4']
    N_NODES = 4

    def __init__(self, cfg, n_pairs=1000, seed=42):
        self.cfg = cfg
        self.n_pairs = n_pairs
        self.rng = np.random.default_rng(seed)
        self.preprocessor = Preprocessor(cfg)
        self.world_weights = ct.utils.dataset_utils.get_world_weights(cfg)
        self.n_worlds = self.world_weights.shape[0]
        self.noise_var = cfg.dataset.world.noise_var
        self.obs_interval = cfg.dataset.observation_interval
        self.int_interval = cfg.dataset.intervention_interval
        self.value_grid = self.preprocessor.values

    def _encode_val(self, value):
        return ct.utils.dataset_utils.encode_value(self.cfg, value, self.preprocessor)

    def _encode_scm(self, scm_index):
        return list(ct.utils.name_utils.encode_scm_index(self.cfg, scm_index))

    def _random_scm_index(self, exclude=None):
        while True:
            idx = int(self.rng.integers(0, self.n_worlds))
            if exclude is None or idx != exclude:
                return idx

    def _random_grid_value(self, interval):
        lo, hi = interval
        valid = self.value_grid[(self.value_grid >= lo) & (self.value_grid <= hi)]
        return float(self.rng.choice(valid))

    def _build_prompt(self, scm_letters, obs_parts, do_parts, query_var):
        tokens = ["INFERENCE"] + scm_letters + obs_parts + do_parts + [query_var]
        return " ".join(tokens)

    def _sample_obs_do(self, scm_index):
        n_obs = int(self.rng.choice(self.cfg.dataset.inference_examples.n_observations))
        n_do = int(self.rng.choice(self.cfg.dataset.inference_examples.n_interventions))
        n_do = max(n_do, 1)
        all_nodes = list(range(self.N_NODES))
        obs_node_indices = list(self.rng.choice(all_nodes, size=min(n_obs, self.N_NODES), replace=False))
        remaining = [n for n in all_nodes if n not in obs_node_indices]
        do_node_indices = list(self.rng.choice(remaining, size=min(n_do, len(remaining)), replace=False))
        obs_vars = [self.VARIABLES[i] for i in obs_node_indices]
        obs_vals = [self._encode_val(self._random_grid_value(self.obs_interval)) for _ in obs_vars]
        do_vars = [self.VARIABLES[i] for i in do_node_indices]
        do_vals = [self._encode_val(self._random_grid_value(self.int_interval)) for _ in do_vars]
        return obs_vars, obs_vals, do_vars, do_vals

    def _query_var(self, do_vars):
        candidates = [v for v in self.VARIABLES if v not in do_vars]
        return str(self.rng.choice(candidates))

    def _make_intervention_pair(self):
        for _ in range(10000):
            scm_index = self._random_scm_index()
            letters = self._encode_scm(scm_index)
            obs_vars, obs_vals, do_vars, do_vals = self._sample_obs_do(scm_index)
            target_slot = self._random_scm_index()
            letters = self._encode_scm(scm_index)
            obs_vars, obs_vals, do_vars, do_vals = self._sample_obs_do(scm_index)
            target_slot = int(self.rng.integers(0, len(do_vars)))
            original_var = do_vars[target_slot]
            forbidden = set(obs_vars) | set(do_vars)
            alternatives = [v for v in self.VARIABLES if v not in forbidden]
            if not alternatives:
                continue
            new_var = str(self.rng.choice(alternatives))
            obs_parts = [f"OBS {v} {val}" for v, val in zip(obs_vars, obs_vals)]
            do_parts_clean = [f"DO {v} {val}" for v, val in zip(do_vars, do_vals)]
            do_parts_corrupt = do_parts_clean.copy()
            do_parts_corrupt[target_slot] = f"DO {new_var} {do_vals[target_slot]}"
            query = self._query_var(do_vars)
            clean = self._build_prompt(letters, obs_parts, do_parts_clean, query)
            corrupt = self._build_prompt(letters, obs_parts, do_parts_corrupt, query)
            return clean, corrupt
        raise RuntimeError("Could not generate a valid intervention pair after 10000 attempts")

    def _make_scm_index_pair(self):
        scm_index = self._random_scm_index()
        letters = self._encode_scm(scm_index)
        obs_vars, obs_vals, do_vars, do_vals = self._sample_obs_do(scm_index)
        corrupt_index = self._random_scm_index(exclude=scm_index)
        corrupt_letters = self._encode_scm(corrupt_index)
        while corrupt_letters == letters:
            corrupt_index = self._random_scm_index(exclude=scm_index)
            corrupt_letters = self._encode_scm(corrupt_index)
        obs_parts = [f"OBS {v} {val}" for v, val in zip(obs_vars, obs_vals)]
        do_parts = [f"DO {v} {val}" for v, val in zip(do_vars, do_vals)]
        query = self._query_var(do_vars)
        clean = self._build_prompt(letters, obs_parts, do_parts, query)
        corrupt = self._build_prompt(corrupt_letters, obs_parts, do_parts, query)
        return clean, corrupt

    def _make_variable_value_pair(self):
        val_clean = self._encode_val(5.0)
        val_corrupt = self._encode_val(-5.0)
        for _ in range(10000):
            scm_index = self._random_scm_index()
            letters = self._encode_scm(scm_index)
            obs_vars, obs_vals, do_vars, do_vals = self._sample_obs_do(scm_index)
            if len(obs_vars) == 0:
                continue
            target_slot = int(self.rng.integers(0, len(obs_vars)))
            obs_vals_clean = obs_vals.copy()
            obs_vals_corrupt = obs_vals.copy()
            obs_vals_clean[target_slot] = val_clean
            obs_vals_corrupt[target_slot] = val_corrupt
            obs_parts_clean = [f"OBS {v} {val}" for v, val in zip(obs_vars, obs_vals_clean)]
            obs_parts_corrupt = [f"OBS {v} {val}" for v, val in zip(obs_vars, obs_vals_corrupt)]
            do_parts = [f"DO {v} {val}" for v, val in zip(do_vars, do_vals)]
            query = self._query_var(do_vars)
            clean = self._build_prompt(letters, obs_parts_clean, do_parts, query)
            corrupt = self._build_prompt(letters, obs_parts_corrupt, do_parts, query)
            return clean, corrupt
        raise RuntimeError("Could not generate a valid variable_value pair after 10000 attempts")

    def generate(self):
        generators = {
            #"intervention": self._make_intervention_pair,
            #"scm_index": self._make_scm_index_pair,
            "variable_value": self._make_variable_value_pair
        }
        datasets = {}
        for name, gen_fn in generators.items():
            pairs = []
            while len(pairs) < self.n_pairs:
                try:
                    pairs.append(gen_fn())
                except RuntimeError as e:
                    print(f"[warn] {name}: {e}")
            datasets[name] = pairs
            print(f"Generated {len(pairs)} pairs for corruption type '{name}'")
        return datasets

    def write(self, datasets, output_dir="."):
        import os
        os.makedirs(output_dir, exist_ok=True)
        filenames = {
            #"intervention": "intervention.txt",
            #"scm_index": "scm_index.txt",
            "variable_value": "variable_value.txt"
        }
        for name, pairs in datasets.items():
            path = os.path.join(output_dir, filenames[name])
            with open(path, "w", encoding="utf-8") as f:
                for clean, corrupt in pairs:
                    f.write(clean + "\n")
                    f.write(corrupt + "\n")
            print(f"Wrote {len(pairs)} pairs -> {path}")

@hydra.main(version_base=None, config_name="config", config_path = CONFIG_PATH)
def generate_patching_datasets(cfg):
    print("Generating activation patching datasets...")
    apd = ActivationPatchingDataset(cfg, n_pairs=1000, seed=42)
    datasets = apd.generate()
    apd.write(datasets, output_dir=".")
    print("Done.")

@hydra.main(version_base=None, config_name="config", config_path=CONFIG_PATH)
def test_causal_dataset_gaussian(cfg):
    """Test function to verify dataset generation"""
    dataset = CausalDatasetGaussian(cfg, split="train")
    print("n data examples: ", dataset.n_data)
    print("n inference examples: ", dataset.n_inference)
    held_out_world_indices = ct.utils.dataset_utils.load_held_out_world_indices(cfg)
    print("held out world indices: ", held_out_world_indices[:10])
    # test a specific example
    idx = 120020
    indices, meta_data = dataset[idx]
    print(meta_data["string"])
    print(meta_data["scm_index"])
    print(dataset.world_weights[meta_data["scm_index"]])
    print()
    # print random examples
    random_indices = np.random.permutation(len(dataset))
    for i in random_indices[:10]:
        indices, meta_data = dataset[i]
        print(meta_data["string"])


if __name__ == "__main__":
    generate_patching_datasets()



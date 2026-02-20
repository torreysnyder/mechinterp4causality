from causal_transformers.utils.name_utils import get_node_name_from_index, encode_scm_index
from causal_transformers.utils.dataset_utils import get_n_ctx
import torch
import torch.nn as nn
import string
import hydra
import causal_transformers as ct
import numpy as np
PAD_TOKEN = 0
EOS_TOKEN = 1


class Preprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vocab = {}
        self.vocab["PAD"] = PAD_TOKEN
        self.vocab["EOS"] = EOS_TOKEN
        self.vocab["DATA"] = 2
        self.vocab["INFERENCE"] = 3
        self.vocab["0"] = 4
        self.vocab["1"] = 5
        self.vocab["OBS"] = 6
        self.vocab["DO"] = 7
        self.num_indices = []

        if cfg.dataset.type == "additive":
            # probabilities 0.00, 0.01, ..., 1.00
            for prob in np.linspace(0, 1, 101):
                prob_str = f"{prob:.2f}"
                self.vocab[prob_str] = len(self.vocab)

        elif cfg.dataset.type == "gaussian":
            # Values on a regular grid between min_num and max_num
            min_num = cfg.dataset.value_encoding.range[0]
            max_num = cfg.dataset.value_encoding.range[1]
            steps = cfg.dataset.value_encoding.steps

            # --- FIX: infer number of decimal places from grid step size ---
            # e.g. range [-5, 5], steps=101 -> step_size=0.1 -> decimal_points=1
            if steps > 1:
                step_size = (max_num - min_num) / (steps - 1)
                if step_size > 0:
                    decimal_points = max(
                        0, int(np.ceil(-np.log10(step_size)))
                    )
                else:
                    decimal_points = 0
            else:
                step_size = 0.0
                decimal_points = 0

            self.decimal_points = decimal_points

            # Create the grid and vocab entries with the inferred precision
            self.values = np.array(np.linspace(min_num, max_num, steps))
            for num in self.values:
                num_str = f"{num:.{decimal_points}f}"
                self.num_indices.append(len(self.vocab))
                self.vocab[num_str] = len(self.vocab)

        else:
            raise Exception(f"Unknown dataset type {cfg.dataset.type}")

        # Add ±INF tokens
        self.vocab["+INF"] = len(self.vocab)
        self.vocab["-INF"] = len(self.vocab)
        self.inf_indices = [len(self.vocab) - 2, len(self.vocab) - 1]

        # Node names
        self.n_nodes_max = cfg.dataset.world.n_nodes.max
        for node in range(self.n_nodes_max):
            node_name = get_node_name_from_index(cfg, node)
            self.vocab[node_name] = len(self.vocab)

        # Capital letters
        for letter in string.ascii_uppercase:
            self.vocab[letter] = len(self.vocab)

        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.n_ctx = get_n_ctx(cfg)

    def tokenize(self, text):
        tokens = text.split()
        return tokens

    def __call__(self, text):
        tokens = self.tokenize(text)

        # Slightly more informative error if something is missing
        try:
            indices = [self.vocab[token] for token in tokens]
        except KeyError as e:
            missing = [t for t in tokens if t not in self.vocab]
            raise KeyError(
                f"Unknown token(s) in Preprocessor vocab: {missing}. "
                "Check dataset.value_encoding range/steps and numeric formatting."
            ) from e

        string_len = len(indices)
        max_len = self.n_ctx + 1
        if string_len > max_len:
            print(
                f"Warning: string of length {string_len} exceeds maximum length "
                f"determined by context of the model {max_len}"
            )
            print("(reducing the size of string to the max len)")
            indices = indices[:max_len]
        indices += [PAD_TOKEN] * (max_len - string_len)
        return torch.tensor(indices, dtype=torch.long)

    def decode_indices(self, indices):
        s = ""
        for index in indices:
            token = self.vocab_inv[index.item()]
            s += token
            if token == "EOS":
                break
            s += " "
        return s


def test_preprocess():
    cfg = ct.utils.config_utils.load_config()
    preprocessor = Preprocessor(cfg)
    print(preprocessor.vocab)
    input_string = "DATA A B E D C X DO OBS V1 0.1 0.2 V4 0.2 V3 0.8 V2 0.1 EOS"
    print("input")
    print(input_string)
    indices = preprocessor(input_string)
    print(indices)
    decoded_string = preprocessor.decode_indices(indices)
    print(decoded_string)

if __name__ == "__main__":
    test_preprocess()


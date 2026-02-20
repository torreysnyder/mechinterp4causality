from numpy import nan
import torch
from causal_transformers.dataset.preprocessor import Preprocessor
from causal_transformers.utils.name_utils import encode_scm_index
import causal_transformers as ct
from typing import Tuple
import numpy as np

class Accuracy:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.preprocessor = ct.dataset.preprocessor.Preprocessor(cfg)
        self.vocab_0 = self.preprocessor.vocab["0"]
        self.vocab_1 = self.preprocessor.vocab["1"]
        self.vocab_D = self.preprocessor.vocab["DATA"]
        self.vocab_I = self.preprocessor.vocab["INFERENCE"]
        self.prob_indices = torch.tensor([self.preprocessor.vocab[f"{prob:.2f}"] for prob in np.linspace(0, 1, 101)], dtype=torch.long).to(device)
        self.held_out_world_indices = ct.utils.dataset_utils.load_held_out_world_indices(cfg)
        self.data_correct_train = 0
        self.data_correct_held = 0
        self.data_count_train = 0
        self.data_count_held = 0
        self.inference_variational_distance_train = 0.0
        self.inference_variational_distance_held = 0.0
        self.inference_count_train = 0
        self.inference_count_held = 0

    def calculate_variational_distance(self, targets, pred_targets, mask):
        target_probs = np.array([float(self.preprocessor.vocab_inv[index.item()]) for index in targets[mask]])
        pred_probs = np.array([self.map_index_to_prob(index.item()) for index in pred_targets[mask]])
        variational_distance = np.abs(target_probs - pred_probs).sum()
        return variational_distance

    def update(
            self,
            logits: torch.Tensor,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            world_indices,
            n_interventions: torch.Tensor
    ):
        pred_targets = torch.argmax(logits, dim=-1)
        data_rows = inputs[:, 0] == self.vocab_D
        train_worlds_mask = torch.isin(world_indices, self.held_out_world_indices, invert=True).to(targets.device)
        if self.cfg.dataset.data_examples.posterior_probs:
            data_mask = torch.isin(targets, self.prob_indices)
        else:
            data_mask = ((targets == self.vocab_0) | (targets == self.vocab_1))
            for i in range(len(n_interventions)):
                true_positions = torch.where(data_mask[i])[0]
                if len(true_positions) > 0:
                    data_mask[i, true_positions[:n_interventions[i]]] = False
        data_mask &= data_rows.unsqueeze(-1)
        data_mask_train = data_mask & train_worlds_mask.unsqueeze(-1)
        data_mask_held = data_mask & ~train_worlds_mask.unsqueeze(-1)
        if self.cfg.dataset.data_examples.posterior_probs:
            self.data_correct_train += self.calculate_variational_distance(targets, pred_targets, data_mask_train)
            self.data_correct_held += self.calculate_variational_distance(targets, pred_targets, data_mask_held)
        else:
            self.data_correct_train += (pred_targets[data_mask_train] == targets[data_mask_train]).sum().item()
            self.data_correct_held += (pred_targets[data_mask_held] == targets[data_mask_held]).sum().item()
        self.data_count_train += data_mask_train.sum().item()
        self.data_count_held += data_mask_held.sum().item()
        inference_mask = torch.isin(targets, self.prob_indices)
        inference_mask &= ~data_rows.unsqueeze(-1)
        inference_mask_train = inference_mask & train_worlds_mask.unsqueeze(-1)
        inference_mask_held = inference_mask & ~train_worlds_mask.unsqueeze(-1)
        self.inference_variational_distance_train += self.calculate_variational_distance(targets, pred_targets, inference_mask_train)
        self.inference_variational_distance_held += self.calculate_variational_distance(targets, pred_targets, inference_mask_held)
        self.inference_count_train += inference_mask_train.sum().item()
        self.inference_count_held += inference_mask_held.sum().item()

    def map_index_to_prob(self, index: int) -> float:
        if index in self.prob_indices:
            return float(self.preprocessor.vocab_inv[index])
        else:
            return 0.50

    def get_data_acc_train(self) -> float:
        if self.cfg.dataset.data_examples.posterior_probs:
            return 1 - 1/2 * self.data_correct_train / self.data_count_train if self.data_count_train > 0 else np.nan
        else:
            return self.data_correct_train / self.data_count_train if self.data_count_train > 0 else np.nan

    def get_data_acc_held(self) -> float:
        if self.cfg.dataset.data_examples.posterior_probs:
            return 1 - 1/2 * self.data_correct_held / self.data_count_held if self.data_count_held > 0 else np.nan
        else:
            return self.data_correct_held / self.data_count_held if self.data_count_held > 0 else np.nan

    def get_inference_acc_train(self) -> float:
        return 1 - 1/2 * self.inference_variational_distance_train / self.inference_count_train if self.inference_count_train > 0 else np.nan

    def get_inference_acc_held(self) -> float:
        return 1 - 1/2 * self.inference_variational_distance_held / self.inference_count_held if self.inference_count_held > 0 else np.nan

from numpy import nan
import torch
from causal_transformers.dataset.preprocessor import Preprocessor
from causal_transformers.utils.name_utils import encode_scm_index
import causal_transformers as ct
from typing import Tuple
import numpy as np

class GaussianAccuracy:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.preprocessor = ct.dataset.preprocessor.Preprocessor(cfg)
        self.vocab_D = self.preprocessor.vocab["DATA"]
        self.vocab_I = self.preprocessor.vocab["INFERENCE"]
        self.num_indices = torch.tensor(self.preprocessor.num_indices).to(device)
        self.inf_indices = torch.tensor(self.preprocessor.inf_indices).to(device)
        self.held_out_world_indices = ct.utils.dataset_utils.load_held_out_world_indices(cfg)
        self.accuracy = {}
        self.count = {}
        for example_type in ["train", "held"]:
            self.accuracy[example_type] = {}
            self.count[example_type] = {}
            for key in ["data", "data_abs", "inference_mean", "inference_mean_abs", "inference_std"]:
                self.accuracy[example_type][key] = 0.0
                self.count[example_type][key] = 0

    def mae(self, targets, pred_targets, stds=True):
        if targets.shape[0] == 0:
            return 0.0, 0.0, 0.0, 0.0
        targets_num = torch.tensor([self.map_index_to_num(index.item()) for index in targets])
        pred_targets_num = torch.tensor([self.map_index_to_num(index.item()) for index in pred_targets])
        if stds:
            targets_num = targets_num.view(-1, 2)
            pred_targets_num = pred_targets_num.view(-1, 2)
            mu1, std1 = targets_num[:, 0], targets_num[:, 1]
            mu2, std2 = pred_targets_num[:, 0], pred_targets_num[:, 1]
            mae_mean = torch.sum(torch.abs(mu1 - mu2))
            mae_mean_abs = torch.sum(torch.abs(torch.abs(mu1) - torch.abs(mu2)))
            mae_std = torch.sum(torch.abs(std1 - std2))
            mae_std_abs = torch.sum(torch.abs(torch.abs(std1) - torch.abs(std2)))
        else:
            mu1 = targets_num
            mu2 = pred_targets_num
            mae_mean = torch.sum(torch.abs(mu1 - mu2))
            mae_mean_abs = torch.sum(torch.abs(torch.abs(mu1) - torch.abs(mu2)))
            mae_std = torch.tensor(0.0)
            mae_std_abs = torch.tensor(0.0)
        if torch.isinf(mae_mean):
            print("infinity encountered")
            print(mu1, mu2, std1, std2)
            mae_mean = torch.tensor(0.0)
            mae_mean_abs = torch.tensor(0.0)
            mae_std = torch.tensor(0.0)
            mae_std_abs = torch.tensor(0.0)
        return mae_mean, mae_mean_abs, mae_std, mae_std_abs

    def update(
            self,
            logits: torch.Tensor,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            meta_data
    ):
        pred_targets = torch.argmax(logits, dim=-1)
        mask = torch.isin(targets, self.num_indices) | torch.isin(targets, self.inf_indices)
        n_interventions = meta_data["n_interventions"]
        n_observations = meta_data["n_observations"]
        world_indices = meta_data["scm_index"]
        for i in range(len(n_interventions)):
            true_positions = torch.where(mask[i])[0]
            if len(true_positions) > 0:
                mask[i, true_positions[:n_interventions[i] + n_observations[i]]] = False
        data_rows = inputs[:, 0] == self.vocab_D
        train_worlds_mask = torch.isin(world_indices, self.held_out_world_indices, invert=True).to(targets.device)
        data_mask = mask.clone()
        data_mask &= data_rows.unsqueeze(-1)
        data_mask_train = data_mask & train_worlds_mask.unsqueeze(-1)
        data_mask_held = data_mask & ~train_worlds_mask.unsqueeze(-1)
        data_mae_train, data_mae_abs_train, _, _ = self.mae(targets[data_mask_train], pred_targets[data_mask_train], stds=False)
        data_mae_held, data_mae_abs_held, _, _ = self.mae(targets[data_mask_held], pred_targets[data_mask_held], stds=False)
        self.accuracy["train"]["data"] += data_mae_train
        self.accuracy["held"]["data"] += data_mae_held
        self.accuracy["train"]["data_abs"] += data_mae_abs_train
        self.accuracy["held"]["data_abs"] += data_mae_abs_held
        self.count["train"]["data"] += data_mask_train.sum().item()
        self.count["held"]["data"] += data_mask_held.sum().item()
        self.count["train"]["data_abs"] += data_mask_train.sum().item()
        self.count["held"]["data_abs"] += data_mask_held.sum().item()
        inference_mask = mask.clone()
        inference_mask &= ~data_rows.unsqueeze(-1)
        inference_mask_train = inference_mask & train_worlds_mask.unsqueeze(-1)
        inference_mask_held = inference_mask & ~train_worlds_mask.unsqueeze(-1)
        inference_mean_train, inference_mean_abs_train, inference_std_train, inference_std_abs_train = self.mae(targets[inference_mask_train], pred_targets[inference_mask_train], stds=True)
        inference_mean_held, inference_mean_abs_held, inference_std_held, inference_std_abs_held = self.mae(targets[inference_mask_held], pred_targets[inference_mask_held], stds=True)
        self.accuracy["train"]["inference_mean"] += inference_mean_train
        self.accuracy["held"]["inference_mean"] += inference_mean_held
        self.accuracy["train"]["inference_mean_abs"] += inference_mean_abs_train
        self.accuracy["held"]["inference_mean_abs"] += inference_mean_abs_held
        self.accuracy["train"]["inference_std"] += inference_std_train
        self.accuracy["held"]["inference_std"] += inference_std_held
        self.count["train"]["inference_mean"] += inference_mask_train.sum().item()
        self.count["held"]["inference_mean"] += inference_mask_held.sum().item()
        self.count["train"]["inference_mean_abs"] += inference_mask_train.sum().item()
        self.count["held"]["inference_mean_abs"] += inference_mask_held.sum().item()
        self.count["train"]["inference_std"] += inference_mask_train.sum().item()
        self.count["held"]["inference_std"] += inference_mask_held.sum().item()

    def map_index_to_num(self, index: int) -> float:
        if index in self.num_indices:
            return float(self.preprocessor.vocab_inv[index])
        elif index in self.inf_indices:
            if index == self.preprocessor.vocab["+INF"]:
                return self.cfg.dataset.value_encoding.range[1]
            else:
                return self.cfg.dataset.value_encoding.range[0]
        else:
            return self.cfg.dataset.value_encoding.range[1]

    def get_accuracy(self):
        acc = {}
        for key in self.accuracy.keys():
            for example_type in self.accuracy[key].keys():
                if self.count[key][example_type] > 0:
                    acc[f"{example_type}_mae_{key}"] = (self.accuracy[key][example_type]/self.count[key][example_type]).item()
                else:
                    acc[f"{example_type}_mae_{key}"] = np.nan
        return acc

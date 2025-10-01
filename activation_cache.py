from __future__ import annotations
import logging
import warnings
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast
import einops
import numpy as np
import torch
from jaxtyping import Float, Int
from typing_extensions import Literal
import utils as utils
from utils import Slice, SliceInput

class ActivationCache:
    def __init__(self, cache_dict: Dict[str, torch.Tensor], model, has_batch_dim: bool = True):
        self.cache_dict = cache_dict
        self.model = model
        self.has_batch_dim = has_batch_dim
        self.has_embed = "hook_embed" in self.cache_dict
        self.has_pos_embed = "hook_pos_embed" in self.cache_dict

    def remove_batch_dim(self) -> ActivationCache:
        if self.has_batch_dim:
            for key in self.cache_dict:
                assert(
                    self.cache_dict[key].size(0) == 1
                ), f"Cannot remove batch dimension from cache with batch size > 1, \ for key {key} with shape {self.cache_dict[key].shape}"
                self.cache_dict[key] = self.cache_dict[key][0]
            self.has_batch_dim = False
        else:
            logging.warning("Tried removing batch dimension after already having removed it.")
        return self

    def __repr__(self) -> str:
        return f"ActivationCache with keys {list(self.cache_dict.keys())}"

    def __getitem__(self, key) -> torch.Tensor:
        if key in self.cache_dict:
            return self.cache_dict[key]
        elif type(key) == str:
            return self.cache_dict[utils.get_act_name(key)]
        else:
            if len(key) > 1 and key[1] is not None:
                if key[1] < 0:
                    key = (key[0], self.model.cfg.n_layers + key[1], *key[2:])
            return self.cache_dict[utils.get_act_name(*key)]

    def __len__(self) -> int:
        return len(self.cache_dict)

    def to(self, device: Union[str, torch.device], move_model=False) -> ActivationCache:
        if move_model is not None:
            warnings.warn(
                "The 'move_model' parameter is deprecated.",
                DeprecationWarning,
            )

        self.cache_dict = {key: value.to(device) for key, value in self.cache_dict.items()}

        if move_model:
            self.model.to(device)

        return self

    def toggle_autodiff(self, mode: bool = False):
        logging.warning("Changed the global state, set autodiff to %s", mode)
        torch.set_grad_enabled(mode)

    def keys(self):
        return self.cache_dict.keys()

    def values(self):
        return self.cache_dict.values()

    def items(self):
        return self.cache_dict.items()

    def __iter__(self) -> Iterator[str]:
        return self.cache_dict.__iter__()

    def apply_slice_to_batch_dim(self, batch_slice: Union[Slice, SliceInput]) -> ActivationCache:
        if not isinstance(batch_slice, Slice):
            batch_slice = Slice(batch_slice)
        batch_slice = cast(Slice, batch_slice)
        assert (
            self.has_batch_dim or batch_slice.mode == "empty"
        ), "Cannot index into a cache without a batch dim"
        still_has_batch_dim = (batch_slice.mode != "int") and self.has_batch_dim
        new_cache_dict = {
            name: batch_slice.apply(param, dim=0) for name, param in self.cache_dict.items()
        }
        return ActivationCache

    def accumulated_resid(
            self,
            layer: Optional[int] = None,
            incl_mid: bool = False,
            apply_ln: bool = False,
            pos_slice: Optional[Union[Slice, SliceInput]] = None,
            mlp_input: bool = False,
            return_labels: bool = False,
    ) -> Union[
        Float[torch.Tensor, "layer_covered *batch_and_pos_dims d_model"],
        Tuple[Float[torch.Tensor, "layers_covered *batch_and_pos_dims d_model"], List[str]],
    ]:
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        if layer is None or layer == -1:
            layer = self.model.cfg.n_layers
        assert isinstance(layer, int)
        labels = []
        components_list = []
        for l in range(layer + 1):
            if l == self.model.cfg.n_layers:
                components_list.append(self[("resid_post", self.model.cfg.n_layers - 1)])
                labels.append("final_post")
                continue
            components_list.append(self[("resid_pre", l)])
            labels.append(f"{l}_pre")
            if (incl_mid and l < layer) or (mlp_input and l == layer):
                components_list.append(self[("resid_mid", l)])
                labels.append(f"{l}_mid")
        components_list = [pos_slice.apply(c, dim=-2) for c in components_list]
        components = torch.stack(components_list, dim=0)
        if apply_ln:
            components = self.apply_ln_to_stack(
                components, layer, pos_slice=pos_slice, mlp_input=mlp_input
            )
        if return_labels:
            return components, labels
        else:
            return components

    def logit_attrs(
            self,
            residual_stack: Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"],
            tokens: Union[
                str,
                int,
                Int[torch.Tensor, ""],
                Int[torch.Tensor, "batch"],
                Int[torch.Tensor, "batch position"],
            ],
            incorrect_tokens: Optional[
                Union[
                    str,
                    int,
                    Int[torch.Tensor, ""],
                    Int[torch.Tensor, "batch"],
                    Int[torch.Tensor, "batch position"],
                ]
            ] = None,
            pos_slice: Union[Slice, SliceInput] = None,
            batch_slice: Union[Slice, SliceInput] = None,
            has_batch_dim: bool = True,
    ) -> Float[torch.Tensor, "num_components *batch_and_pos_dims_out"]:
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        if not isinstance(batch_slice, Slice):
            batch_slice = Slice(batch_slice)
        if isinstance(tokens, str):
            tokens = torch.as_tensor(self.model.to_single_token(tokens))
        elif isinstance(tokens, int):
            tokens = torch.as_tensor(tokens)
        logit_directions = self.model.tokens_to_residual_directions(tokens)
        if incorrect_tokens is not None:
            if isinstance(incorrect_tokens, str):
                incorrect_tokens = torch.as_tensor(self.model.to_single_token(incorrect_tokens))
            elif isinstance(incorrect_tokens, int):
                incorrect_tokens = torch.as_tensor(incorrect_tokens)
            if tokens.shape != incorrect_tokens.shape:
                raise ValueError(
                    f"tokens and incorrect_tokens must have the same shape! (tokens.shape={tokens.shape}, incorrect_tokens.shape={incorrect_tokens.shape})"
                )
            logit_directions = logit_directions - self.model.tokens_to_residual_directions(incorrect_tokens)
        scaled_residual_stack = self.apply_ln_to_stack(
            residual_stack,
            layer=-1,
            pos_slice=pos_slice,
            batch_slice=batch_slice,
            has_batch_dim=has_batch_dim,
        )
        logit_attrs = (scaled_residual_stack * logit_directions).sum(dim=-1)
        return logit_attrs

    def decompose_resid(
            self,
            layer: Optional[int] = None,
            mlp_input: bool = False,
            mode: Literal["all", "mlp", "attn"] = "all",
            apply_ln: bool = False,
            pos_slice: Union[Slice, SliceInput] = None,
            incl_embeds: bool = True,
            return_labels: bool = False,
    ) -> Union[
        Float[torch.Tensor, "layers_covered *batch_and_pos_dims d_model"],
        Tuple[Float[torch.Tensor, "layers_covered *batch_and_pos_dims d_model"], List[str]],
    ]:
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        pos_slice = cast(Slice, pos_slice)
        if layer is None or layer == -1:
            layer = self.model.cfg.n_layers
        assert isinstance(layer, int)

        incl_attn = mode != "mlp"
        incl_mlp = mode != "attn" and not self.model.cfg.attn_only
        components_list = []
        labels = []
        if incl_embeds:
            if self.has_embed:
                components_list = [self["hook_embed"]]
                labels.append("embed")
            if self.has_pos_embed:
                components_list.append(self["hook_pos_embed"])
                labels.append("pos_embed")
        for l in range(layer):
            if incl_attn:
                components_list.append(self[("attn_out", l)])
                labels.append(f"{l}_attn_out")
            if incl_mlp:
                components_list.append(self[("mlp_out", l)])
                labels.append(f"{l}_mlp_out")
        if mlp_input and incl_attn:
            components_list.append(self[("attn_out", layer)])
            labels.append(f"{layer}_attn_out")
        components_list = [pos_slice.apply(c, dim=-2) for c in components_list]
        components = torch.stack(components_list, dim=0)
        if apply_ln:
            components = self.apply_ln_to_stack(
                components, layer, pos_slice=pos_slice, mlp_input=mlp_input
            )
        if return_labels:
            return components, labels
        else:
            return components

    def compute_head_results(self):
        if "blocks.0.attn.hook_result" in self.cache_dict:
            logging.warning("Tried to compute head results when they were already cached")
            return
        for layer in range(self.model.cfg.n_layers):
            z = einops.rearrange(self[("z", layer, "attn")],
                                 "... head_index d_head -> ... head_index d_head 1",
            )
            result = z * self.model.blocks[layer].attn.W_O
            self.cache_dict[f"blocks.{layer}.attn.hook_result"] = result.sum(dim=-2)

    def stack_head_results(
            self,
            layer: int = -1,
            return_labels: bool = False,
            incl_remainder: bool = False,
            pos_slice: Union[Slice, SliceInput] = None,
            apply_ln: bool = False,
    ) -> Union[
        Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"],
        Tuple[Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"], List[str]]
    ]:
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        pos_slice = cast(Slice, pos_slice)
        if layer is None or layer == -1:
            layer = self.model.cfg.n_layers
        if "blocks.0.attn.hook_result" not in self.cache_dict:
            print(
                "Tried to stack head results when they weren't cached. Computing head results now"
            )
            self.compute_head_results()
        components: Any = []
        labels = []
        for l in range(layer):
            components.append(pos_slice.apply(self[("result", l, "attn")], dim=-3))
            labels.extend([f"L{l}H{h}" for h in range(self.model.cfg.n_heads)])
        if components:
            components = torch.cat(components, dim=-2)
            components = einops.rearrange(
                components,
                "... concat_head_index d_model -> concat_head_index ... d_model"
            )
            if incl_remainder:
                remainder = pos_slice.apply(
                    self[("resid_post", layer - 1)], dim=-2
                ) - components.sum(dim=0)
                components = torch.cat([components, remainder[None]], dim=0)
                labels.append("remainder")
        elif incl_remainder:
            components = torch.cat(
                [pos_slice.apply(self[("resid_post", layer - 1)], dim=-2)[None]], dim=0
            )
            labels.append("remainder")
        else:
            components = torch.zeros(
                0,
                *pos_slice.apply(self["hook_embed"], dim=-2).shape,
                device=self.model.cfg.device,
            )
        if apply_ln:
            components = self.apply_ln_to_stack(components, layer, pos_slice=pos_slice)
        if return_labels:
            return components, labels
        else:
            return components

    def stack_activation(
            self,
            activation_name: str,
            layer: int = -1,
            sublayer_type: Optional[str] = None,
    ) -> Float[torch.Tensor, "layers_covered ..."]:
        if layer is None or layer == -1:
            layer = self.model.cfg.n_layers
        components = []
        for l in range(layer):
            components.append(self[(activation_name, l, sublayer_type)])
        return torch.stack(components, dim=0)

    def get_neuron_results(
            self,
            layer: int,
            neuron_slice: Union[Slice, SliceInput] = None,
            pos_slice: Union[Slice, SliceInput] = None,
    ) -> Float[torch.Tensor, "*batch_and_pos_dims num_neurons d_model"]:
        if not isinstance(neuron_slice, Slice):
            neuron_slice = Slice(neuron_slice)
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        neuron_acts = self[("post", layer, "mlp")]
        W_out = self.model.blcoks[layer].mlp.W_out
        if pos_slice is not None:
            neuron_acts = pos_slice.apply(neuron_acts, dim=-2)
        if neuron_slice is not None:
            neuron_acts = neuron_slice.apply(neuron_acts, dim=-1)
            W_out = neuron_slice.apply(W_out, dim=0)
        return neuron_acts[..., None] * W_out

    def stack_neuron_results(
            self,
            layer: int,
            pos_slice: Union[Slice, SliceInput] = None,
            neuron_slice: Union[Slice, SliceInput] = None,
            return_labels: bool = False,
            incl_remainder: bool = False,
            apply_ln: bool = False,
    ) -> Union[
        Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"],
        Tuple[Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"], List[str]],
    ]:
        if layer is None or layer == -1:
            layer = self.model.cfg.n_layers
        components: Any = []
        labels = []
        if not isinstance(neuron_slice, Slice):
            neuron_slice = Slice(neuron_slice)
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        neuron_labels: Union[torch.Tensor, np.ndarray] = neuron_slice.apply(
            torch.arange(self.model.cfg.d_mlp), dim=0
        )
        if isinstance(neuron_labels, int):
            neuron_labels = np.array([neuron_labels])
        for l in range(layer):
            components.append(
                self.get_neuron_results(l, pos_slice=pos_slice, neuron_slice=neuron_slice)
            )
            labels.extend([f"L{l}N{h}" for h in neuron_labels])
        if components:
            components = torch.cat(components, dim=-2)
            components = einops.rearrante(
                components,
                "... concat_neuron_index d_model -> concat_neuron_index ... d_model",
            )
            if incl_remainder:
                remainder = pos_slice.apply(
                    self[("resid_post", layer - 1)], dim=-2
                ) - components.sum(dim=0)
                components = torch.cat([components, remainder[None]], dim=0)
                labels.append("remainder")
        elif incl_remainder:
            components = torch.cat(
                [pos_slice.apply(self[("resid_post", layer - 1)], dim=-2)[None]], dim=0
            )
            labels.append("remainder")
        else:
            components = torch.zeros(
                0,
                *pos_slice.apply(self["hook_embed"], dim=-2).shape,
                device=self.model.cfg.device,
            )
        if apply_ln:
            components = self.apply_ln_to_stack(components, layer, pos_slice=pos_slice)
        if return_labels:
            return components, labels
        else:
            return components

    def apply_ln_to_stack(
            self,
            residual_stack: Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"],
            layer: Optional[int] = None,
            mlp_input: bool = False,
            pos_slice: Union[Slice, SliceInput] = None,
            batch_slice: Union[Slice, SliceInput] = None,
            has_batch_dim: bool = True,
    ) -> Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"]:
        if self.model.cfg.normalization_type not in ["LN", "LNPre", "RMS", "RMSPre"]:
            return residual_stack
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        if not isinstance(batch_slice, Slice):
            batch_slice = Slice(batch_slice)

        if layer is None or layer == -1:
            layer = self.model.cfg.n_layers
        if has_batch_dim:
            residual_stack = batch_slice.apply(residual_stack, dim=1)
        if self.model.cfg.noramlization_type in ["LN", "LNPre"]:
            residual_stack = residual_stack = residual_stack.mean(dim=-1, keepdim=True)
        if layer == self.model.cfg.n_layers or layer is None:
            scale = self["ln_final.hook_scale"]
        else:
            hook_name = f"blocks.{layer}.ln{2 if mlp_input else 1}.hook_scale"
            scale = self[hook_name]
        scale = pos_slice.apply(scale, dim=-2)
        if self.has_batch_dim:
            scale = batch_slice.apply(scale)
        return residual_stack / scale

    def get_full_resid_decomposition(
            self,
            layer: Optional[int] = None,
            mlp_input: bool = False,
            expand_neurons: bool = True,
            apply_ln: bool = False,
            pos_slice: Union[Slice, SliceInput] = None,
            return_labels: bool = False,
    ) -> Union[
        Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"],
        Tuple[Float[torch.Tensor, "num_components *batch_and_pos_dims d_model"], List[str]],
    ]:
        if layer is None or layer == -1:
            layer = self.model.cfg.n_layers
        assert layer is not None
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        head_stack, head_labels = self.stack_head_results(
            layer + (1 if mlp_input else 0), pos_slice=pos_slice, return_labels=True
        )
        labels = head_labels
        components = [head_stack]
        if not self.model.cfg.attn_only and layer > 0:
            if expand_neurons:
                neuron_stack, neuron_labels = self.stack_neuron_results(
                    layer, pos_slice=pos_slice, return_labels=True
                )
                labels.extend(neuron_labels)
                components.append(neuron_stack)
            else:
                mlp_stack, mlp_labels = self.decompose_resid(
                    layer,
                    mlp_input=mlp_input,
                    pos_slice=pos_slice,
                    incl_embeds=False,
                    mode="mlp",
                    return_labels=True,
                )
                labels.extend(mlp_labels)
                components.append(mlp_stack)
        if self.has_embed:
            labels.append("embed")
            components.append(pos_slice.apply(self["embed"], -2)[None])
        if self.has_pos_embed:
            labels.append("pos_embed")
            components.append(pos_slice.apply(self["pos_embed"], -2)[None])
        bias = self.model.accumulated_bias(layer, mlp_input, include_mlp_biases=expand_neurons)
        bias = bias.expand((1,) + head_stack.shape[1:])
        labels.append("bias")
        components.append(bias)
        residual_stack = torch.cat(components, dim=0)
        if apply_ln:
            residual_stack = self.apply_ln_to_stack(
                residual_stack, layer, pos_slice=pos_slice, mlp_input=mlp_input
            )
        if return_labels:
            return residual_stack, labels
        else:
            return residual_stack


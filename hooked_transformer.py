from __future__ import annotations
import logging
import os
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload
)

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm.auto as tqdm
from jaxtyping import Float, Int
from packaging import version
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing_extensions import Literal

import loading_from_pretrained as loading
import utils as utils
from activation_cache import ActivationCache
from components.embed import Embed
from components.layer_norm import LayerNorm
from components.layer_norm_pre import LayerNormPre
from components.pos_embed import PosEmbed
from components.rms_norm import RMSNorm
from components.rms_norm_pre import RMSNormPre
from components.transformer_block import TransformerBlock
from components.unembed import Unembed


from factored_matrix import FactoredMatrix
from hook_points import HookedRootModule, HookPoint
from hooked_tranformer_config import HookedTransformerConfig
from loading_from_pretrained import NON_HF_HOSTED_MODEL_NAMES

from past_key_value_caching import HookedTransformerKeyValueCache
from utilities import devices
from utils import (
    USE_DEFAULT_VALUE,
    init_kaiming_normal_,
    init_kaiming_uniform_,
    init_xavier_normal_,
    init_xavier_uniform_,
)

SingleLoss = Float[torch.Tensor, ""]
LossPerToken = Float[torch.Tensor, "batch pos-1"]
Loss = Union[SingleLoss, LossPerToken]

DTYPE_FROM_STRING = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

T = TypeVar("T", bound="HookedTransformer")

class Output(NamedTuple):
    logits: Float[torch.Tensor, "batch pos d_vocab"]
    loss: Loss

class HookedTransformer(HookedRootModule):
    ln_final: nn.Module
    tokenizer: Optional[PreTrainedTokenizerBase]
    def __init__(
            self,
            cfg: Union[HookedTransformerConfig, Dict],
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            move_to_device: bool = True,
            default_padding_side: Literal["left", "right"] = "right",
    ):
        super().__init__()
        if isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedTransformerConfig object. If you want to load a pretrained model, use HookedTransformer.from_pretrained() instead."
            )
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        if tokenizer is not None:
            self.set_tokenizer(tokenizer, default_padding_side=default_padding_side)
        elif self.cfg.tokenizer_name is not None:
            if self.cfg.tokenizer_name in NON_HF_HOSTED_MODEL_NAMES:
                logging.warning(
                    "%s tokenizer not loaded. Please load manually.",
                    self.cfg.tokenizer_name,
                )
            else:
                use_fast = True
                if "phi" in self.cfg.tokenizer_name.lower():
                    use_fast = False
                huggingface_token = os.environ.get("HF_TOKEN", "")
                self.set_tokenizer(
                    AutoTokenizer.from_pretrained(
                        self.cfg.tokenizer_name,
                        add_bos_token=True,
                        trust_remote_code=self.cfg.trust_remote_code,
                        use_fast=use_fast,
                        token=huggingface_token if len(huggingface_token) > 0 else None,
                    ),
                    default_padding_side=default_padding_side,
                )
        else:
            assert self.cfg.d_vocab != -1, "Must provide a tokenizer if d_vocab is not provided"
            self.tokenizer = None
            if default_padding_side != "right":
                logging.warning(
                    "default_padding_side is explicitly given but ignored because tokenizer is not set"
                )
        self.embed = Embed(self.cfg)
        self.hook_embed = HookPoint()
        if self.cfg.positional_embedding_type != "rotary":
            self.pos_embed = PosEmbed(self.cfg)
            self.hook_pos_embed = HookPoint()
        if self.cfg.use_hook_tokens:
            self.hook_tokens = HookPoint()
        self.blocks = nn.ModuleList(
            [TransformerBlock(self.cfg, block_index) for block_index in range(self.cfg.n_layers)]
        )
        if self.cfg.normalization_type == "RMS":
            self.ln_final = RMSNorm(self.cfg)
        elif self.cfg.normalization_type == "RMSPre":
            self.ln_final = RMSNormPre(self.cfg)
        elif self.cfg.normalization_type == "LN":
            if self.cfg.final_rms:
                self.ln_final = RMSNorm(self.cfg)
            else:
                self.ln_final = LayerNorm(self.cfg)
        elif self.cfg.normalization_type == "LNPre":
            if self.cfg_final_rms:
                self.ln_final = RMSNormPre(self.cfg)
            else:
                self.ln_final = LayerNormPre(self.cfg)
        elif self.cfg.normalization_type is None:
            pass
        else:
            logging.warning("Invalid normalization_type passed in %s", self.cfg.normalization_type)
        self.unembed = Unembed(self.cfg)
        if self.cfg.init_weights:
            self.init_weights()
        if move_to_device:
            self.move_model_modules_to_device()
        self.dataset = None
        self.setup()

    def check_hooks_to_add(
            self,
            hook_point,
            hook_point_name,
            hook,
            dir="fwd",
            is_permanent=False,
            prepend=False
    ) -> None:
        if hook_point_name.endswith("attn.hook_result"):
            assert (
                self.cfg.use_attn_result
            ), f"Cannot add hook {hook_point_name} is use_attn_result_hook is False"
        if hook_point_name.endswith(("hook_q_input", "hook_k_input", "hook_v_input")):
            assert (
                self.cfg.use_split_qkv_input
            ), f"Cannot add hook {hook_point_name} is use_split_qkv_input is False"
        if hook_point_name.endswith("mlp_in"):
            assert (
                self.cfg.use_hook_mlp_in
            ), f"Cannot add hook {hook_point_name} if use_hook_mlp_in is False"
        if hook_point_name.endswith("attn_in"):
            assert (
                self.cfg.use_attn_in
            ), f"Cannot add hook {hook_point_name} is use_attn_in is False"

    def get_pos_offset(self, past_kv_cache, batch_size):
        if past_kv_cache is None:
            pos_offset = 0
        else:
            (
                cached_batch_size,
                cache_ctx_length,
                num_heads_in_cache,
                d_head_in_cache,
            ) = past_kv_cache[0].past_keys.shape
            assert cached_batch_size == batch_size
            if self.cfg.n_key_value_heads is None:
                assert num_heads_in_cache == self.cfg.n_heads
            else:
                assert num_heads_in_cache == self.cfg.n_key_value_heads
            assert d_head_in_cache == self.cfg.d_head
            pos_offset = cache_ctx_length
        return pos_offset

    def get_residual(
            self,
            embed,
            pos_offset,
            prepend_bos=USE_DEFAULT_VALUE,
            attention_mask=None,
            tokens=None,
            return_shortformer_pos_embed=True,
            device=None,
    ):
        if device is None:
            device = devices.get_device_for_block_index(0, self.cfg)
        if tokens is None:
            tokens = torch.ones((embed.size(0), embed.size(1))).int().to(device)
        if self.cfg.positional_embedding_type == "standard":
            pos_embed = self.hook_pos_embed(
                self.pos_embed(tokens, pos_offset, attention_mask)
            )
            residual = embed + pos_embed
            shortformer_pos_embed = None
        elif self.cfg.positional_embedding_type == "shortformer":
            pos_embed = self.hook_pos_embed(
                self.pos_embed(tokens, pos_offset, attention_mask)
            )
            residual = embed
            shortformer_pos_embed = pos_embed
        elif self.cfg.positional_embedding_type == "rotary":
            residual = embed
            shortformer_pos_embed = None
        elif self.cfg.positional_embedding_type == "alibi":
            residual = embed
            shortformer_pos_embed = None
        else:
            raise ValueError(
                f"Invalid positional_embedding_type passed in {self.cfg.positional_embedding_type}"
            )
        if return_shortformer_pos_embed:
            return residual, shortformer_pos_embed
        else:
            return residual

    def input_to_embed(
            self,
            input: Union[str, List[str], Int[torch.Tensor, "batch pos"]],
            prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
            padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
            attention_mask: Optional[torch.Tensor] = None,
            past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> Tuple[
        Float[torch.Tensor, "batch pos d_model"],
        Optional[Int[torch.Tensor, "batch pos"]],
        Optional[Float[torch.Tensor, "batch pos d_model"]],
        Optional[torch.Tensor],
    ]:
        if isinstance(input, str) or isinstance(input, list):
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if passing a string to the model"
            tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
        else:
            tokens = input
        if len(tokens.shape) == 1:
            tokens = tokens[None]
        if tokens.device.type != self.cfg.device:
            tokens = tokens.to(devices.get_device_for_block_index(0, self.cfg))
        if (
                (self.tokenizer and self.tokenizer.padding_side == "left")
            or attention_mask is not None
            or past_kv_cache is not None
        ):
            if attention_mask is None:
                if prepend_bos is USE_DEFAULT_VALUE:
                    prepend_bos = self.cfg.default_prepend_bos
                if self.tokenizer is None:
                    raise ValueError("Cannot compute attention mask without a tokenizer.")
                attention_mask = utils.get_attention_mask(self.tokenizer, tokens, prepend_bos)
            assert attention_mask.shape == tokens.shape, (
                f"Attention mask shape {attention_mask.shape} does not match tokens shape {tokens.shape}"
            )
            attention_mask = attention_mask.to(devices.get_device_for_block_index(0, self.cfg))
            if past_kv_cache is not None:
                attention_mask = past_kv_cache.append_attention_mask(attention_mask)
        else:
            attention_mask = None
        batch_size = tokens.shape[0]
        pos_offset = self.get_pos_offset(past_kv_cache, batch_size)
        if self.cfg.use_hook_tokens:
            tokens = self.hook_tokens(tokens)
        embed = self.hook_embed(self.embed(tokens))
        residual, shortformer_pos_embed = self.get_residual(
            embed,
            pos_offset,
            prepend_bos,
            attention_mask,
            tokens,
            return_shortformer_pos_embed=True,
        )
        return residual, tokens, shortformer_pos_embed, attention_mask

    @overload
    def forward(
            self,
            input,
            return_type: Literal["logits"],
            loss_per_token: bool = False,
            prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
            padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
            start_at_layer: Optional[int] = None,
            tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
            shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            stop_at_layer: Optional[int] = None,
            past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> Loss:
        ...

    @overload
    def forward(
            self,
            input,
            return_type: Literal["loss"],
            loss_per_token: bool = False,
            prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
            padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
            start_at_layer: Optional[int] = None,
            tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
            shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            stop_at_layer: Optional[int] = None,
            past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> Loss:
        ...

    @overload
    def forward(
        self,
        input,
        return_type: Literal["loss"],
        loss_per_token: bool = False,
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
        start_at_layer: Optional[int] = None,
        tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
        shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        stop_at_layer: Optional[int] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> Loss:
        ...

    @overload
    def forward(
            self,
            input,
            return_type: Literal["loss"],
            loss_per_token: bool = False,
            prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
            padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
            start_at_layer: Optional[int] = None,
            tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
            shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            stop_at_layer: Optional[int] = None,
            past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> Loss:
        ...

    @overload
    def forward(
            self,
            input,
            return_type: Literal["both"],
            loss_per_token: bool = False,
            prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
            padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
            start_at_layer: Optional[int] = None,
            shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            stop_at_layer: Optional[int] = None,
            past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss]:
        ...

    @overload
    def forward(
            self,
            input,
            return_type: Literal[None],
            loss_per_token: bool = False,
            prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
            padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
            start_at_layer: Optional[int] = None,
            tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
            shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            stop_at_layer: Optional[int] = None,
            past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> None:
        ...

    def forward(
            self,
            input: Union[
                str,
                List[str],
                Int[torch.Tensor, "batch pos"],
                Float[torch.Tensor, "batch pos d_model"],
            ],
            return_type: Optional[str] = "logits",
            loss_per_token: bool = False,
            prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
            padding_side: Optional[Literal["left", "right"]] = USE_DEFAULT_VALUE,
            start_at_layer: Optional[int] = None,
            tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
            shortformer_pos_embed: Optional[Float[torch.Tensor, "batch pos d_model"]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            stop_at_layer: Optional[int] = None,
            past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> Union[
        None,
        Float[torch.Tensor, "batch pos d_vocab"],
        Loss,
        Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
    ]:
        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            if start_at_layer is None:
                (
                    residual,
                    tokens,
                    shortformer_pos_embed,
                    attention_mask,
                ) = self.input_to_embed(
                    input,
                    prepend_bos=prepend_bos,
                    padding_side=padding_side,
                    attention_mask=attention_mask,
                    past_kv_cache=past_kv_cache,
                )
            else:
                assert type(input) == torch.Tensor
                residual = input
            if start_at_layer is None:
                start_at_layer = 0
            blocks_and_idxs = list(zip(range(self.cfg.n_layers), self.blocks))
            for i, block in blocks_and_idxs[start_at_layer:stop_at_layer]:
                residual = residual.to(devices.get_device_for_block_index(i, self.cfg))
                if shortformer_pos_embed is not None:
                    shortformer_pos_embed = shortformer_pos_embed.to(
                        devices.get_device_for_block_index(i, self.cfg)
                    )
                residual = block(
                    residual,
                    past_kv_cache_entry=past_kv_cache[i] if past_kv_cache is not None else None,
                    shortformer_pos_embed=shortformer_pos_embed,
                    attention_mask=attention_mask,
                )
            if stop_at_layer is not None:
                return residual
            if self.cfg.normalization_type is not None:
                residual = self.ln_final(residual)
            if return_type is None:
                return None
            else:
                logits = self.unembed(residual)
                if self.cfg.output_logits_soft_cap > 0.0:
                    logits = self.cfg.output_logits_soft_cap * F.tanh(logits/self.cfg.output_logits_soft_cap)
                if return_type == "logits":
                    return logits
                else:
                    assert(
                        tokens is not None
                    ), "tokens must be passed in if return_type is 'loss' or 'both'"
                    loss = self.loss_fn(logits, tokens, attention_mask, per_token=loss_per_token)
                    if return_type == "loss":
                        return loss
                    elif return_type == "both":
                        return Output(logits, loss)
                    else:
                        logging.warning(f"Invalid return_type passed in: {return_type}")
                        return None

    def loss_fn(
            self,
            logits: Float[torch.Tensor, "batch pos d_vocab"],
            tokens: Int[torch.Tensor, "batch pos"],
            attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
            per_token: bool = False,
    ):
        if tokens.device != logits.device:
            tokens = tokens.to(logits.device)
        return utils.lm_cross_entropy_loss(logits, tokens, attention_mask, per_token)

    @overload
    def run_with_cache(
            self, *model_args, return_cache_object: Literal[False], **kwargs
    ) -> Tuple[Output, ActivationCache]:
        ...

    @overload
    def run_with_cache(
            self, *model_args, return_cache_object: Literal[False], **kwargs
    ) -> Tuple[Output, Dict[str, torch.Tensor]]:
        ...

    def run_with_cache(
            self, *model_args, return_cache_object=True, remove_batch_dim=False, **kwargs
    ) -> Tuple[
        Union[
            None,
            Float[torch.Tensor, "batch pos d_vocab"],
            Loss,
            Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
        ],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        if return_cache_object:
            cache = ActivationCache(cache_dict, self, has_batch_dim=not remove_batch_dim)
            return out, cache
        else:
            return out, cache_dict

    def set_tokenizer(
            self,
            tokenizer,
            default_padding_side="right",
    ):
        assert isinstance(
            tokenizer, PreTrainedTokenizerBase
        ), f"{type(tokenizer)} is not a supported tokenizer, please use PreTrainedTokenizer or PreTrainedTokenizerFast"
        assert default_padding_side in [
            "right",
            "left",
        ], f"padding_side must be 'right' or 'left', got {default_padding_side}"

        tokenizer_with_bos = utils.get_tokenizer_with_bos(tokenizer)
        self.tokenizer = tokenizer_with_bos
        self.tokenizer.padding_side = default_padding_side
        self.cfg.tokenizer_prepend_bos = len(self.tokenizer.encode("")) > 0
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "<|endoftext|>"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

        if self.cfg.d_vocab == -1:
            self.cfg.d_vocab = max(self.tokenizer.vocab.values()) + 1
        if self.cfg.d_vocab_out == -1:
            self.cfg.d_vocab_out = self.cfg.d_vocab

    def to_tokens(
            self,
            input: Union[str, List[str]],
            prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
            padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
            move_to_device: bool = True,
            truncate: bool = True,
    ) -> Int[torch.Tensor, "batch pos"]:
        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            assert self.tokenizer is not None, "Cannot use to_tokens without a tokenizer"
            assert (
                self.cfg.tokenizer_prepends_bos is not None
            ), "Set the tokenizer for the model by calling set_tokenizer"

            if self.cfg.default_prepend_bos and not self.cfg.tokenizer_prepends_bos:
                input = utils.get_input_with_manually_prepended_bos(self.tokenizer, input)
            tokens = self.tokenizer(
                input,
                return_tesnors="pt",
                padding=True,
                truncation=truncate,
                max_length=self.cfg.n_ctx if truncate else None,
            )["input_ids"]
            if not self.cfg.default_prepend_bos and self.cfg.tokenizer_prepends_bos:
                tokens = utils.get_tokens_with_bos_removed(self.tokenizer, tokens)
            if move_to_device:
                tokens = tokens.to(self.cfg.device)
            return tokens

    def to_string(
            self,
            tokens: Union[
                List[int],
                Int[torch.Tensor, ""],
                Int[torch.Tensor, "batch pos"],
                Int[torch.Tensor, "pos"],
                np.ndarray,
                List[Int[torch.Tensor, "pos"]]
            ],
    ) -> Union[str, List[str]]:
        assert self.tokenizer is not None, "Cannot use to_string without a tokenizer"
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens)
        if len(tokens.shape) == 2:
            return self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
        elif len(tokens.shape) <= 1:
            return self.tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
        else:
            raise ValueError(f"Invalid shape passsed in {tokens.shape}")

    def to_str_tokens(
            self,
            input: Union[
                str,
                Int[torch.Tensor, "pos"],
                Int[torch.Tensor, "1 pos"],
                Int[np.ndarray, "pos"],
                Int[np.ndarray, "1 pos"],
                list,
            ],
            prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
            padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
    ) -> Union[List[str], List[List[str]]]:
        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            assert self.tokenizer is not None
            tokens: Union[np.ndarray, torch.Tensor]
            if isinstance(input, list):
                return list(
                    map(
                        lambda tokens: self.to_str_tokens(tokens, prepend_bos, padding_side),
                        input,
                    )
                )
            elif isinstance(input, str):
                tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)[0]
                if "gemma" in self.tokenizer.name_or_path and tokens.ndim == 1:
                    tokens = tokens.unsqueeze(1)
            elif isinstance(input, torch.Tensor):
                tokens = input
                tokens = tokens.squeeze()
                if tokens.dim() == 0:
                    tokens = tokens.unsqueeze(0)
                assert (
                    tokens.dim() == 1
                ), f"Invalid tokens input to to_str_tokens, has shape: {tokens.shape}"
            elif isinstance(input, np.ndarray):
                tokens = input
                tokens = tokens.squeeze()
                if tokens.ndim == 0:
                    tokens = np.expand_dims(tokens, axis=0)
                assert (
                    tokens.ndim == 1
                ), f"Invalid tokens input to to_str_tokens, has shape: {tokens.shape}"
            else:
                raise ValueError(f"Invalid input type to to_str_tokens: {type(input)}")
            str_tokens = self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
            return str_tokens

    def to_single_token(self, string):
        token = self.to_tokens(string, prepend_bos=False).squeeze()
        assert not token.shape, f"Input string: {string} is not a single token!"
        return token.item()

    def to_single_str_token(self, int_token: int) -> str:
        assert isinstance(int_token, int)
        token = self.to_str_tokens(torch.tensor([int_token]))
        assert len(token) == 1
        return cast(str, token[0])

    def get_token_position(
            self,
            single_token: Union[str, int],
            input: Union[str, Union[Float[torch.Tensor, "pos"], Float[torch.Tensor, "1 pos"]]],
            mode="first",
            prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
            padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
    ):
        if isinstance(input, str):
            tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
        else:
            tokens = input

        if len(tokens.shape) == 2:
            assert (
                tokens.shape[0] == 1
            ), f"If tokens are rank two, they must have shape [1, seq_len], not {tokens.shape}"
            tokens = tokens[0]

        if isinstance(single_token, str):
            single_token = self.to_single_token(single_token)
        elif isinstance(single_token, torch.Tensor):
            single_token = single_token.item()

        indices = torch.arange(len(tokens), device=tokens.device)[tokens == single_token]
        assert len(indices) > 0, "The token does not occur in the prompt"
        if mode == "first":
            return indices[0].item()
        elif mode == "last":
            return indices[-1].item()
        else:
            raise ValueError(f"mode must be 'first' or 'last', not {mode}")

    def tokens_to_residual_directions(
            self,
            tokens: Union[
                str,
                int,
                Int[torch.Tensor, ""],
                Int[torch.Tensor, "pos"],
                Int[torch.Tensor, "batch pos"],
            ],
    ) -> Union[
        Float[torch.Tensor, "d_model"],
        Float[torch.Tensor, "pos d_model"],
        Float[torch.Tensor, "batch pos d_model"],
    ]:
        if isinstance(tokens, torch.Tensor) and tokens.numel() > 1:
            residual_directions = self.W_U[:, tokens]
            residual_directions = einops.rearrange(
                residual_directions, "d_model ... -> ... d_model"
            )
            return residual_directions
        else:
            if isinstance(tokens, str):
                token = self.to_single_token(tokens)
            elif isinstance(tokens, int):
                token = tokens
            elif isinstance(tokens, torch.Tensor) and tokens.numel() == 1:
                token = tokens.item()
            else:
                raise ValueError(f"Invalid token type: {type(tokens)}")
            residual_direction = self.W_U[:, token]
            return residual_direction

    def to(
            self,
            device_or_dtype: Union[torch.device, str, torch.dtype],
            print_details: bool = True
    ):
        return devices.move_to_and_update_config(self, device_or_dtype, print_details)

    def cuda(self: T, device: Optional[Union[int, torch.device]] = None) -> T:
        if isinstance(device, int):
            return self.to(f"cuda: {device}")
        elif device is None:
            return self.to("cuda")
        else:
            return self.to(device)

    def cpu(self: T) -> T:
        return self.to(torch.device("cpu"))

    def mps(self: T) -> T:
        return self.to(torch.device("mps"))

    def move_model_modules_to_device(self):
        self.embed.to(devices.get_best_available_device(self.cfg))
        self.hook_embed.to(devices.get_best_available_device(self.cfg))
        if self.cfg.positional_embedding_type != "rotary":
            self.pos_embed.to(devices.get_best_available_device(self.cfg))
            self.hook_pos_embed.to(devices.get_best_available_device(self.cfg))
        if hasattr(self, "ln_final"):
            self.ln_final.to(devices.get_best_available_device(self.cfg))
        self.unembed.to(devices.get_best_available_device(self.cfg))
        for i, block in enumerate(self.blocks):
            block.to(devices.get_best_available_device(self.cfg))

    @classmethod
    def from_pretrained(
            cls: Type[T],
            model_name: str,
            fold_ln: bool = True,
            center_writing_weights: bool = True,
            center_unembed: bool = True,
            refactor_factored_attn_matrices: bool = False,
            checkpoint_index: Optional[int] = None,
            checkpoint_value: Optional[int] = None,
            hf_model: Optional[Any] = None,
            device: Optional[Union[str, torch.device]] = None,
            n_devices: int = 1,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            move_to_device: bool = True,
            fold_value_biases: bool = True,
            default_prepend_bos: Optional[bool] = None,
            default_padding_side: Literal["left", "right"] = "right",
            dtype="float32",
            first_n_layers: Optional[int] = None,
            **from_pretrained_kwargs,
    ) -> T:
        if model_name.lower().startswith("t5"):
            raise RuntimeError(
                "Execution stopped: Please use HookedEncoderDecoder to load T5 models instead of HookedTransformer"
            )
        assert not(
            from_pretrained_kwargs.get("load_in_8bit", False)
            or from_pretrained_kwargs.get("load_in_4bit", False)
        ), "Quantization not supported"

        if hf_model is not None:
            assert hf_model.config is not None
            hf_cfg = hf_model.config.to_dict()
            qc = hf_cfg.get("quantization_config", {})
            load_in_4bit = qc.get("load_in_4bit", False)
            load_in_8bit = qc.get("load_in_8bit", False)
            quant_method = qc.get("quant_method", "")
            assert not load_in_8bit, "8-bit quantization is not supported"
            assert not (
                load_in_4bit and (version.parse(torch.__version__) < version.parse("2.1.1"))
            ), "Quantization is only supported for torch versions >= 2.1.1"
            assert not (
                load_in_4bit and ("llama" not in model_name.lower())
            ), "Quantization is only supported for Llama models"
            if load_in_4bit:
                assert (
                    qc.get("quant_method", "") == "bitsandbytes"
                ), "Only bitsandbytes quantization is supported"
        else:
            hf_cfg = {}

        if isinstance(dtype, str):
            dtype = DTYPE_FROM_STRING[dtype]
        if "torch_dtype" in from_pretrained_kwargs:
            dtype = from_pretrained_kwargs["torch_dtype"]
        if (
            (from_pretrained_kwargs.get("torch_dtype", None) == torch.float16)
            or dtype == torch.float16
        ) and device in ["cpu", None]:
            logging.warning("float16 models may not work on CPU. Consider using a GPU or bfloat16.")
        official_model_name = loading.get_official_model_name(model_name)
        cfg = loading.get_pretrained_model_config(
            official_model_name,
            hf_cfg=hf_cfg,
            checkpoint_index=checkpoint_index,
            checkpoint_value=checkpoint_value,
            fold_ln=fold_ln,
            device=device,
            n_devices=n_devices,
            default_prepend_bos=default_prepend_bos,
            dtype=dtype,
            first_n_layers=first_n_layers,
            **from_pretrained_kwargs,
        )
        if cfg.positional_embedding_type == "shortformer":
            if fold_ln:
                logging.warning(
                    "You tried to specify fold_ln=True for a shortformer model, but this can't be done! Setting fold_ln=False instead."
                )
                fold_ln = False
            if center_unembed:
                logging.warning(
                    "You tried to specify center_unembed=True for a shortformer model, but this can't be done! Settign center_unembed=False instead."
                )
                center_unembed = False
            if center_writing_weights:
                logging.warning(
                    "You tried to specify center_writing_weights=True for a shortformer model, but this can't be done! Setting center_writing_weights=False instead."
                )
                center_writing_weights = False
        if center_unembed and cfg.output_logits_soft_cap > 0.0:
            logging.warning(
                "You tried to specify center_unembed=True for a model using logit softcap, but this can't be done! Softcapping is not invariant upon adding a constant. Setting center_unembed=False instead"
            )
            center_unembed = False
        state_dict = loading.get_pretrained_state_dict(
            official_model_name, cfg, hf_model, dtype=dtype, **from_pretrained_kwargs
        )
        model = cls(
            cfg,
            tokenizer,
            move_to_device=False,
            default_padding_side=default_padding_side,
        )
        model.load_and_process_state_dict(
            state_dict,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            fold_value_biases=fold_value_biases,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
        )
        if move_to_device:
            model.move_model_modules_to_device()
        print(f"Loaded pretrained model {model_name} into HookedTransformer")
        return model

    @classmethod
    def from_pretrained_no_processing(
            cls,
            model_name: str,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            refactor_factored_attn_matrices=False,
            fold_value_biases=False,
            dtype=torch.float32,
            default_prepend_bos=None,
            default_padding_side="right",
            **from_pretrained_kwargs,
    ):
        return cls.from_pretrained(
            model_name,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            fold_value_biases=fold_value_biases,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
            dtype=dtype,
            default_prepend_bos=default_prepend_bos,
            default_padding_side=default_padding_side,
            **from_pretrained_kwargs,
        )

    def init_weights(self):
        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
        if self.cfg.init_mode == "gpt2":
            self._init_weights_gpt2()
        elif self.cfg.init_mode == "xavier_uniform":
            self._init_weights_xavier(dist_type="uniform")
        elif self.cfg.init_mode == "xavier_normal":
            self._init_weights_xavier(dist_type="normal")
        elif self.cfg.init_mode == "kaiming_uniform":
            self._init_weights_kaiming(dist_type="uniform")
        elif self.cfg.init_mode == "kaiming_normal":
            self._init_weights_kaiming(dist_type="normal")
        elif self.cfg.init_mode == "muP":
            self._init_weights_muP(dist_type="normal")

    def _init_weights_gpt2(self):
        for name, param in self.named_parameters():
            if "W_" in name:
                nn.init.normal_(param, std=self.cfg.initializer_range)

    def _init_weights_xavier(self, dist_type="normal"):
        gain = self.cfg.initializer_range
        for name, param in self.named_parameters():
            if "W_" in name:
                if dist_type == "uniform":
                    init_xavier_uniform_(param, gain=gain)
                elif dist_type == "normal":
                    init_xavier_normal_(param, gain=gain)

    def _init_weights_kaiming(self, dist_type="uniform"):
        gain = self.cfg.initializer_range
        for name, param in self.named_parameters():
            if "W_" in name:
                if dist_type == "uniform":
                    init_kaiming_uniform_(param, gain=gain, nonlinearity="relu", mode="fan_in")
                elif dist_type == "normal":
                    init_kaiming_normal_(param, gain=gain, nonlinearity="relu", mode="fan_in")

    def _init_weights_muP(self, dist_type="uniform"):
        for name, param in self.named_parameters():
            if "W_" in name:
                fan_in, _ = utils.calc_fan_in_and_fan_out(param)
                if "embed" in name:
                    scale = float(1)
                elif "unembed" in name:
                    scale = 1/fan_in
                else:
                    scale = 1/fan_in*0.5
                if dist_type == "uniform":
                    scale *= 3**0.5
                    nn.init.uniform_(param, -scale, scale)
                elif dist_type == "normal":
                    nn.init.normal_(param, std=scale)

    def load_and_process_state_dict(
            self,
            state_dict: Dict[str, torch.Tensor],
            fold_ln: bool = True,
            center_writing_weights: bool = True,
            center_unembed: bool = True,
            fold_value_biases: bool = True,
            refactor_factored_attn_matrices: bool = False,
    ):
        if self.cfg.dtype not in [torch.float32, torch.float64] and fold_ln:
            logging.warning(
                "With reduced precision, it is advised to use 'from pretrained_no_processing' instead of 'from_pretrained'."
            )
        if (
            self.cfg.dtype not in [torch.float32, torch.float64]
            and self.cfg.num_experts
            and self.cfg.num_experts > 1
        ):
            logging.warning(
                "When running MoE models, it is advised to use a higher precision data type. See docs for more info."
            )
        state_dict = self.fill_missing_keys(state_dict)
        if fold_ln:
            if self.cfg.num_experts and self.cfg.num_experts > 1:
                logging.warning(
                    "You are using MoE, so the layer norm weights can't be folded! Skipping"
                )
            elif self.cfg.normalization_type in ["LN", "LNPre"]:
                state_dict = self.fold_layer_norm(state_dict)
            elif self.cfg.normalization_type in ["RMS", "RMSPre"]:
                state_dict = self.fold_layer_norm(
                    state_dict, fold_biases=False, center_weights=False
                )
            else:
                logging.warning(
                    "You are not using LayerNorm or RMSNorm, so the layer norm weights can't be folded! Skipping"
                )
        if center_writing_weights:
            if self.cfg.normalization_type not in ["LN", "LNPre"]:
                logging.warning(
                    "You are not using LayerNorm, so the writing weights can't be centered! Skipping"
                )
            elif self.cfg.final_rms:
                logging.warning(
                    "This model is using final RMS normalization, so the writing weights can't be centered! Skipping"
                )
            else:
                state_dict = self.center_writing_weights(state_dict)
        if center_unembed:
            state_dict = self.center_unembed(state_dict)
        if fold_value_biases:
            state_dict = self.fold_value_biases(state_dict)
        if refactor_factored_attn_matrices:
            state_dict = self.refactor_factored_attn_matrices(state_dict)
        if self.cfg.load_in_4bit:
            self.load_state_dict(state_dict, assign=True, strict=False)
        else:
            state_dict_keys = list(state_dict.keys())
            for key in state_dict_keys:
                self.load_state_dict({key: state_dict[key]}, strict=False)
                del state_dict[key]

    def fill_missing_keys(self, state_dict):
        return loading.fill_missing_keys(self, state_dict)

    def fold_layer_norm(
            self, state_dict: Dict[str, torch.Tensor], fold_biases=True, center_weights=True
    ):
        gqa = "" if self.cfg.n_key_value_heads is None else "_"
        for l in range(self.cfg.n_layers):
            if fold_biases:
                state_dict[f"blocks.{l}.attn.b_Q"] = state_dict[f"blocks.{l}.attn.b_Q"] + (
                    state_dict[f"blocks.{l}.attn.W_Q"]
                    * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
                ).sum(-2)
                state_dict[f"blocks.{l}.attn.{gqa}b_K"] = state_dict[
                    f"blocks.{l}.attn.{gqa}b_K"
                ] + (
                    state_dict[f"blocks.{l}.attn.{gqa}W_K"]
                    * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
                ).sum(-2)
                state_dict[f"blocks.{l}.attn.{gqa}b_V"] = state_dict[
                    f"blocks.{l}.attn.{gqa}b_V"
                ] + (
                    state_dict[f"blocks.{l}.attn.{gqa}W_V"]
                    * state_dict[f"blocks.{l}.ln1.b"][None, :, None]
                ).sum(-2)
                del state_dict[f"blocks.{l}.ln1.b"]
            state_dict[f"blocks.{l}.attn.W_Q"] = (
                state_dict[f"blocks.{l}.attn.W_Q"] * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )
            state_dict[f"blocks.{l}.attn.{gqa}W_K"] = (
                state_dict[f"blocks.{l}.attn.{gqa}W_K"]
                * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )
            state_dict[f"blocks.{l}.attn.{gqa}W_V"] = (
                state_dict[f"blocks.{l}.attn.{gqa}W_V"]
                * state_dict[f"blocks.{l}.ln1.w"][None, :, None]
            )
            del state_dict[f"blocks.{l}.ln1.w"]
            if center_weights:
                state_dict[f"blocks.{l}.attn.W_Q"] -= einops.reduce(
                    state_dict[f"blocks.{l}.attn.W_Q"],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )
                state_dict[f"blocks.{l}.attn.{gqa}W_K"] -= einops.reduce(
                    state_dict[f"blocks.{l}.attn.{gqa}W_K"],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )
                state_dict[f"blocks.{l}.attn{gqa}W_V"] -= einops.reduce(
                    state_dict[f"blocks.{l}.attn.{gqa}W_V"],
                    "head_index d_model d_head -> head_index 1 d_head",
                    "mean",
                )
            if not self.cfg.attn_only:
                if fold_biases:
                    state_dict[f"blocks.{l}.mlp.b_in"] = state_dict[f"blocks.{l}.mlp.b_in"] + (
                        state_dict[f"blcoks.{l}.mlp.W_in"]
                        * state_dict[f"blocks.{l}.ln2.b"][:, None]
                    ).sum(-2)
                    del state_dict[f"blocks.{l}.ln2.b"]
                state_dict[f"blocks.{l}.mlp.W_in"] = (
                    state_dict[f"blocks.{l}.mlp.W_in"] * state_dict[f"blocks.{l}.ln2.w"][:, None]
                )
                if self.cfg.gated_mlp:
                    state_dict[f"blocks.{l}.mlp.W_gate"] = (
                        state_dict[f"blocks.{l}.mlp.W_gate"]
                        * state_dict[f"blocks.{l}.ln2.w"][:, None]
                    )
                del state_dict[f"blocks.{l}.ln2.w"]
                if center_weights:
                    state_dict[f"blocks.{l}.mlp.W_in"] -= einops.reduce(
                        state_dict[f"blocks.{l}.mlp.W_in"],
                        "d_model d_mlp -> 1 d_mlp",
                        "mean",
                    )
                if self.cfg.act_fn is not None and self.cfg.act_fn.startswith("solu"):
                    if fold_biases:
                        state_dict[f"blocks.{l}.mlp.b_out"] = state_dict[f"blocks.{l}.mlp.b_out"]
                        + (
                            state_dict[f"blocks.{l}.mlp.W_out"] * state_dict[f"blocks.{l}.mlp.ln.b"][:, None]
                        ).sum(-2)
                        del state_dict[f"blocks.{l}.mlp.ln.b"]
                    state_dict[f"blocks.{l}.mlp.W_out"] = (
                        state_dict[f"blocks.{l}.mlp.W_out"]
                        * state_dict[f"blocks.{l}.mlp.ln.w"][:, None]
                    )
                    if center_weights:
                        state_dict[f"blocks.{l}.mlp.W_out"] -= einops.reduce(
                            state_dict[f"blocks.{l}.mlp.W_out"],
                            "d_mlp d_model -> 1 d_model",
                            "mean",
                        )
                    del state_dict[f"blocks.{l}.mlp.ln.w"]
        if not self.cfg.final_rms and fold_biases:
            state_dict[f"unembed.b_U"] = state_dict[f"unembed.b_U"] + (
                state_dict[f"unembed.W_U"] * state_dict[f"ln_final.b"][:, None]
            ).sum(dim=-2)
            del state_dict[f"ln_final.b"]
        state_dict[f"unembed.W_U"] = state_dict[f"unembed.W_U"] * state_dict[f"ln_final.w"][:, None]
        del state_dict[f"ln_final.w"]
        if center_weights:
            state_dict[f"unembed.W_U"] -= einops.reduce(
                state_dict[f"unembed.W_U"], "d_model d_vocab -> 1 d_vocab", "mean"
            )
        return state_dict

    def center_writing_weights(self, state_dict: Dict[str, torch.Tensor]):
        state_dict["embed.W_E"] = state_dict["embed.W_E"] - state_dict["embed.W_E"].mean(-1, keepdim=True)
        if self.cfg.positional_embedding_type != "rotary":
            state_dict["pos_embed.W_pos"] = state_dict["pos_embed.W_pos"] - state_dict["pos_embed.W_pos"].mean(-1, keepdim=True)
        for l in range(self.cfg.n_layers):
            state_dict[f"blocks.{l}.attn.W_O"] = state_dict[f"blocks.{l}.attn.W_O"] - state_dict[f"blocks.{l}.attn.W_O"].mean(-1, keepdim=True)
            state_dict[f"blocks.{l}.attn.b_O"] = (
                state_dict[f"blocks.{l}.attn.b_O"] - state_dict[f"blocks.{l}.attn.b_O"].mean()
            )
            if not self.cfg.attn_only:
                state_dict[f"blocks.{l}.mlp.W_out"] = state_dict[f"blocks.{l}.mlp.W_out"] - state_dict[f"blocks.{l}.mlp.b_out"].mean(-1, keepdim=True)
                state_dict[f"blocks.{l}.mlp.b_out"] = (state_dict[f"blocks.{l}.mlp.b_out"] - state_dict[f"blocks.{l}.mlp.b_out"].mean()
                )
        return state_dict

    def center_unembed(self, state_dict: Dict[str, torch.Tensor]):
        state_dict["unembed.W_U"] = state_dict["unembed.W_U"] - state_dict["unembed.W_U"].mean(-1, keepdim=True)
        state_dict["unembed.b_U"] = state_dict["unembed.b_U"] - state_dict["unembed.b_U"].mean()
        return state_dict

    def fold_value_biases(self, state_dict: Dict[str, torch.Tensor]):
        for layer in range(self.cfg.n_layers):
            if self.cfg.n_key_value_heads is None:
                b_V = state_dict[f"blocks.{layer}.attn.b_V"]
            else:
                b_V = state_dict[f"blocks.{layer}.attn._b_V"]
                b_V = torch.repeat_interleave(
                    b_V, dim=0, repeats=self.cfg.n_heads // self.cfg.n_key_value_heads
                )
            W_O = state_dict[f"blocks.{layer}.attn.W_O"]
            b_O_original = state_dict[f"blocks.{layer}.attn.b_O"]
            folded_b_O = b_O_original + (b_V[:, :, None] * W_O).sum([0, 1])
            state_dict[f"blocks.{layer}.attn.b_O"] = folded_b_O
            if self.cfg.n_key_value_heads is None:
                state_dict[f"blocks.{layer}.attn.b_V"] = torch.zeros_like(b_V)
            else:
                state_dict[f"blocks.{layer}.attn._b_V"] = torch.zeros_like(state_dict[f"blocks.{layer}.attn._b_V"])
        return state_dict

    def refactor_factored_attn_matrices(self, state_dict: Dict[str, torch.Tensor]):
        assert (
            self.cfg.positional_embedding_type != "rotary"
        ), "You can't refactor the QK circuit when using rotary emebddings (as the QK matrix depends on the position of the query and key)"
        for l in range(self.cfg.n_layers):
            W_Q_eff = torch.cat(
                [
                    state_dict[f"blocks.{l}.attn.W_Q"],
                    state_dict[f"blocks.{l}.attn.b_Q"][:, None, :],
                ],
                dim=1,
            )
            W_K_eff = torch.cat(
                [
                    state_dict[f"blocks.{l}.attn.W_K"],
                    state_dict[f"blocks.{l}.attn.b_K"][:, None, :],
                ],
                dim=1,
            )
            W_Q_eff_even, W_K_eff_even_T = (
                FactoredMatrix(W_Q_eff, W_K_eff.transpose(-1, -2)).make_even().pair
            )
            W_K_eff_even = W_K_eff_even_T.transpose(-1, -2)
            state_dict[f"blocks.{l}.attn.W_Q"] = W_Q_eff_even[:, :-1, :]
            state_dict[f"blocks.{l}.attn.b_Q"] = W_Q_eff_even[:, -1, :]
            state_dict[f"blocks.{l}.attn.W_K"] = W_K_eff_even[:, :-1, :]
            state_dict[f"blocks.{l}.attn.b_K"] = W_K_eff_even[:, -1, :]
            W_V = state_dict[f"blocks.{l}.attn.W_V"]
            W_O = state_dict[f"blocks.{l}.attn.W_O"]
            b_V = state_dict[f"blocks.{l}.attn.b_V"]
            b_O = state_dict[f"blocks.{l}.attn.b_O"]
            b_V_expanded = einops.rearrange(b_V, "head_index d_head -> head_index d_head 1")
            b_V_times_W_O = b_V_expanded * W_O
            b_V_contribution = b_V_times_W_O.sum(1).sum(0)
            effective_bias = b_O + b_V_contribution
            state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros_like(b_V)
            state_dict[f"blocks.{l}.attn.b_O"] = effective_bias
            W_OV = FactoredMatrix(W_V, W_O)
            U, S, Vh = W_OV.svd()
            state_dict[f"blocks.{l}.attn.W_V"] = U @ S.diag_embed()
            state_dict[f"blocks.{l}.attn.W_O"] = utils.tranpose(Vh)
        return state_dict

    def set_use_attn_result(self, use_attn_result: bool):
        self.cfg.use_attn_result = use_attn_result

    def set_use_split_qkv_input(self, use_split_qkv_input: bool):
        self.cfg.use_split_qkv_input = use_split_qkv_input

    def set_use_hook_mlp_in(self, use_hook_mlp_in: bool):
        assert not self.cfg.attn_only, "can't use hook mlp_in with attn_only model"
        self.cfg.use_hook_mlp_in = use_hook_mlp_in

    def set_use_attn_in(self, use_attn_in: bool):
        assert(
            self.cfg.n_key_value_heads is None
        ), "Can't use attn_in with GroupedQueryAttention, please use split_qkv_input instead"
        self.cfg.use_attn_in = use_attn_in

    def set_ungroup_grouped_query_attention(self, ungroup_grouped_query_attention: bool):
        self.cfg.ungroup_grouped_query_attention = ungroup_grouped_query_attention

    def process_weights_(
            self,
            fold_ln: bool = True,
            center_writing_weights: bool = True,
            center_unembed: bool = True,
            refactor_factored_attn_matrices: bool = False,
    ):
        state_dict = self.state_dict()
        if fold_ln and self.cfg.num_experts and self.cfg.num_experts > 1:
            pass
        elif fold_ln and self.cfg.normalization_type == "LN":
            self.cfg.normalization_type = "LNPre"
            self.ln_final = LayerNormPre(self.cfg)
            for layer in self.blocks:
                layer.ln1 = LayerNormPre(self.cfg)
                layer.ln2 = LayerNormPre(self.cfg)
                if self.cfg.is_layer_norm_activation():
                    layer.mlp.ln = LayerNormPre(self.cfg)
        elif fold_ln and self.cfg.normalization_type == "RMS":
            self.cfg.normalization_type = "RMSPre"
            self.ln_final = RMSNormPre(self.cfg)
            for layer in self.blocks:
                layer.ln1 = RMSNormPre(self.cfg)
                layer.ln2 = RMSNormPre(self.cfg)
                if self.cfg.is_layer_norm_activation():
                    layer.mlp.ln = RMSNormPre(self.cfg)
        self.load_and_process_state_dict(
            state_dict,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
        )

    @torch.inference_mode()
    def generate(
            self,
            input: Union[
                str,
                List[str],
                Int[torch.Tensor, "batch pos"],
                Float[torch.Tensor, "batch pos hidden_size"],
            ] = "",
            max_new_tokens: int = 10,
            stop_at_eos: bool = True,
            eos_token_id: Optional[int] = None,
            do_sample: bool = True,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            temperature: float = 1.0,
            freq_penalty: float = 0.0,
            use_past_kv_cache: bool = True,
            prepend_bos: Optional[bool] = USE_DEFAULT_VALUE,
            padding_side: Optional[Literal["left", "right"]] = USE_DEFAULT_VALUE,
            return_type: Optional[str] = "input",
            verbose: bool = True,
    ) -> Union[
        str,
        List[str],
        Int[torch.Tensor, "batch pos_plus_new_tokens"],
        Float[torch.Tensor, "batch pos_plus_new_tokens hidden_size"],
    ]:
        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            assert isinstance(input, (str, torch.Tensor, list)) and (
                isinstance(input, list)
                and all(isinstance(i, str) for i in input)
                or not isinstance(input, list)
            ), "Input must be either string, torch.Tensor, or List[str]"
            assert return_type in [
                "input",
                "str",
                "tokens",
                "embeds",
            ], "return_type must be one of ['input', 'str', 'tokens', 'embeds']"
            if return_type == "input":
                if isinstance(input, (str, list)):
                    return_type = "str"
                elif input.ndim == 2:
                    return_type = "tokens"
                else:
                    return_type = "embeds"
            if isinstance(input, (str, list)):
                input_type = "str"
                assert (
                    self.tokenizer is not None
                ), "Must provide a tokenizer if passing a string to the model"
                input = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
            elif input.ndim == 2:
                input_type = "tokens"
            else:
                input_type = "embeds"
            input_tokens = input if input_type in ["str", "tokens"] else None
            batch_size, ctx_length = input.shape[0], input.shape[1]
            device = devices.get_device_for_block_index(0, self.cfg)
            input = input.to(device)
            if use_past_kv_cache:
                past_kv_cache = HookedTransformerKeyValueCache.init_cache(
                    self.cfg, self.cfg.device, batch_size
                )
            else:
                past_kv_cache = None
            shortformer_pos_embed = None
            embeds = input if input_type == "embeds" else self.embed(input)
            assert isinstance(embeds, torch.Tensor) and embeds.ndim == 3
            stop_tokens: List[int] = []
            eos_token_for_padding = 0
            assert self.tokenizer is not None
            if stop_at_eos:
                tokenizer_has_eos_token = (
                    self.tokenizer is not None and self.tokenizer.eos_token_id is not None
                )
                if eos_token_id is None:
                    assert (
                        tokenizer_has_eos_token
                    ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"
                    eos_token_id = self.tokenizer.eos_token_id
                if isinstance(eos_token_id, int):
                    stop_tokens = [eos_token_id]
                    eos_token_for_padding = eos_token_id
                else:
                    stop_tokens = eos_token_id
                    eos_token_for_padding = (
                        self.tokenizer.eos_token_id if tokenizer_has_eos_token else eos_token_id[0]
                    )
            finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.cfg.device)
            self.eval()
            sampled_tokens_list: List[torch.Tensor] = []
            for index in tqdm.tqdm(range(max_new_tokens), disable=not verbose):
                pos_offset = self.get_pos_offset(past_kv_cache, batch_size)
                tokens = torch.zeros((embeds.size(0), embeds.size(1))).to(torch.int)
                attention_mask = utils.get_attention_mask(
                    self.tokenizer, tokens, False if prepend_bos is None else prepend_bos
                ).to(device)
                residual, shortformer_pos_embed = self.get_residual(
                    embeds, pos_offset, return_shortformer_pos_embed=True, device=device, attention_mask=attention_mask,
                )
                start_at_layer = 0
                if use_past_kv_cache:
                    if index > 0:
                        logits = self.forward(
                            residual[:, -1:],
                            return_type="logits",
                            prepend_bos=prepend_bos,
                            padding_side=padding_side,
                            past_kv_cache=past_kv_cache,
                            start_at_layer=start_at_layer,
                            shortformer_pos_embed=shortformer_pos_embed,
                        )
                    else:
                        logits = self.forward(
                            residual,
                            return_type="logits",
                            prepend_bos=prepend_bos,
                            padding_side=padding_side,
                            past_kv_cache=past_kv_cache,
                            start_at_layer=start_at_layer,
                            shortformer_pos_embed=shortformer_pos_embed,
                        )
                else:
                    logits = self.forward(
                        residual,
                        return_type="logits",
                        prepend_bos=prepend_bos,
                        padding_side=padding_side,
                        start_at_layer=start_at_layer,
                        shortformer_pos_embed=shortformer_pos_embed,
                    )
                final_logits = logits[:, -1, :]
                if do_sample:
                    if input_type in [
                        "str",
                        "tokens",
                    ]:
                        assert input_tokens is not None
                        sampled_tokens = utils.sample_logits(
                            final_logits,
                            top_k=top_k,
                            top_p=top_p,
                            temperature=temperature,
                            freq_penalty=freq_penalty,
                            tokens=torch.cat(
                                (input_tokens, torch.cat(sampled_tokens_list, dim=1)), dim=1
                            )
                           if "sampled_tokens" in locals()
                            else input_tokens,
                        ).to(devices.get_device_for_block_index(0, self.cfg))
                    else:
                        sampled_tokens = utils.sample_logits(
                            final_logits, top_k=top_k, top_p=top_p, temperature=temperature
                        ).to(devices.get_devicie_for_block_index(0, self.cfg))
                else:
                    sampled_tokens = final_logits.argmax(-1).to(
                        devices.get_device_for_block_index(0, self.cfg)
                    )
                sampled_tokens_list.append(sampled_tokens.unsqueeze(1))
                if stop_at_eos:
                    sampled_tokens[finished_sequences] = eos_token_for_padding
                    finished_sequences.logical_or_(
                        torch.isin(
                            sampled_tokens.to(self.cfg.device),
                            torch.tensor(stop_tokens).to(self.cfg.device),
                        )
                    )
                embeds = torch.hstack([embeds, self.embed(sampled_tokens.unsqueeze(-1))])
                if stop_at_eos and finished_sequences.all():
                    break
            sampled_tokens = torch.cat(sampled_tokens_list, dim=1)
            if input_type in ["str", "tokens"]:
                assert input_tokens is not None
                output_tokens = torch.cat((input_tokens, sampled_tokens), dim=1)
            else:
                output_tokens = sampled_tokens
            if return_type == "str":
                decoded_texts = [
                    self.tokenizer.decode(tokens, skip_special_tokens=True)
                    for tokens in output_tokens
                ]
                return decoded_texts[0] if len(decoded_texts) == 1 else decoded_texts
            elif return_type == "tokens":
                return output_tokens
            else:
                return embeds

    @property
    def W_U(self) -> Float[torch.Tensor, "d_model d_vocab"]:
        return self.unembed.W_U

    @property
    def b_U(self) -> Float[torch.Tensor, "d_vocab"]:
        return self.unembed.b_U

    @property
    def W_E(self) -> Float[torch.Tensor, "d_vocab d_model"]:
        return self.embed.W_E

    @property
    def W_pos(self) -> Float[torch.Tensor, "n_ctx d_model"]:
        return self.pos_embed.W_pos

    @property
    def W_E_pos(self) -> Float[torch.Tensor, "d_vocab+n_ctx d_model"]:
        return torch.cat([self.W_E, self.W_pos], dim=0)

    @property
    def W_K(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        return torch.stack([block.attn.W_K for block in self.blocks], dim=0)

    @property
    def W_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        return torch.stack([block.attn.W_Q for block in self.blocks], dim=0)

    @property
    def W_V(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head d_model"]:
        return torch.stack([block.attn.W_V for block in self.blocks], dim=0)

    @property
    def W_O(self) -> Float[torch.Tensor, "n_layers d_model d_mlp"]:
        return torch.stack([block.mlp.W_O for block in self.blocks], dim=0)

    @property
    def W_in(self) -> Float[torch.Tensor, "n_layers d_model d_mlp"]:
        return torch.stack([block.mlp.W_in for block in self.blocks], dim=0)

    @property
    def W_gate(self) -> Union[Float[torch.Tensor, "n_layers d_model d_mlp"], None]:
        if self.cfg.gated_mlp:
            return torch.stack([block.mlp.W_gate for block in self.blocks], dim=0)
        else:
            return None

    @property
    def W_out(self) -> Float[torch.Tensor, "n_layers d_mlp d_model"]:
        return torch.stack([block.mlp.W_out for block in self.blocks], dim=0)

    @property
    def b_K(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        return torch.stack([block.attn.b_K for block in self.blocks], dim=0)

    @property
    def b_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        return torch.stack([block.attn.b_Q for block in self.blocks], dim=0)

    @property
    def b_V(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        return torch.stack([block.attn.b_V for block in self.blocks], dim=0)

    @property
    def b_O(self) -> Float[torch.Tensor, "n_layers d_model"]:
        return torch.stack([block.attn.b_O for block in self.blocks], dim=0)

    @property
    def b_in(self) -> Float[torch.Tensor, "n_layers d_mlp"]:
        return torch.stack([block.mlp.b_in for block in self.blocks], dim=0)

    @property
    def b_out(self) -> Float[torch.Tensor, "n_layers d_model"]:
        return torch.stack([block.mlp.b_out for block in self.blocks], dim=0)

    @property
    def QK(self):
        return FactoredMatrix(self.W_Q, self.W_K.transpose(-2, -1))

    @property
    def OV(self):
        return FactoredMatrix(self.W_V, self.W_O)

    def accumulated_bias(
            self, layer: int, mlp_input: bool = False, include_mlp_biases=True
    ) -> Float[torch.Tensor, "d_model"]:
        accumulated_bias = torch.zeros(self.cfg.d_model, device=self.cfg.device)
        for i in range(layer):
            block = cast(TransformerBlock, self.blocks[i])
            accumulated_bias += cast(torch.Tensor, block.attn.b_O)
            if include_mlp_biases:
                accumulated_bias += cast(torch.Tensor, block.mlp.b_out)
        if mlp_input:
            assert layer < self.cfg.n_layers, "Cannot include attn_bias from beyond the final layer"
            block = cast(TransformerBlock, self.blocks[layer])
            accumulated_bias += cast(torch.Tensor, block.attn.b_O)
        return accumulated_bias

    def all_composition_scores(
            self, mode
    ) -> Float[torch.Tensor, "n_layers n_heads n_layers n_heads"]:
        left = self.OV
        if mode == "Q":
            right = self.QK
        elif mode == "K":
            right = self.QK.T
        elif mode == "V":
            right = self.OV
        else:
            raise ValueError(f"mode must be one of ['Q', 'K', 'V'] not {mode}")
        scores = utils.composition_scores(left, right, broadcast_dims=True)
        mask = (
            torch.arange(self.cfg.n_layers, device=self.cfg.device)[:, None, None, None]
            < torch.arange(self.cfg.n_layers, device=self.cfg.device)[None, None, :, None]
        )
        scores = torch.where(mask, scores, torch.zeros_like(scores))
        return scores

    def all_head_labels(self):
        return [f"L{l}H{h}" for l in range(self.cfg.n_layers) for h in range(self.cfg.n_heads)]

    def load_sample_training_dataset(self, **kwargs):
        model_dataset_map = {
            "neel": "c4_code",
            "neel-solu-old": "pile",
            "GPT2LMHeadModel": "openwebtext",
            "GPTNeoforCausalLM": "pile",
            "GPTNeoXforCausalLM": "pile",
            "GPTJForCausalLM": "pile",
            "OPTForCausalLM": "pile",
        }
        if self.cfg.original_architecture in model_dataset_map:
            self.dataset = utils.get_dataset(
                model_dataset_map[self.cfg.original_architecture], **kwargs
            )
        else:
            raise ValueError(f"We do not have an available dataset for the relevant model: {self.cfg.original_architecture}")
        return self.dataset

    def sample_datapoint(
            self,
            tokenize: bool = False,
            prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
            padding_side: Optional[Literal["left", "right"]] = USE_DEFAULT_VALUE,
    ) -> Union[str, Float[torch.Tensor, "1 pos"]]:
        if self.dataset is None:
            self.load_sample_training_dataset()
        assert self.dataset is not None
        sample_dataset_size = len(self.dataset)
        index = np.random.randint(0, sample_dataset_size)
        if not tokenize:
            return self.dataset[index]["text"]
        else:
            return self.to_tokens(
                self.dataset[index]["text"],
                prepend_bos=prepend_bos,
                padding_side=padding_side,
                truncate=True,
            )


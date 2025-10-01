from __future__ import annotations
import collections.abc
import inspect
import json
import os
import re
import shutil
from copy import deepcopy
from typing import Any, List, Optional, Tuple, Union, cast
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset
from huggingface_hub import constants, hf_hub_download
from jaxtyping import Float, Int
from rich import print as rprint
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from factored_matrix import FactoredMatrix

# Cache directory for Hugging Face downloads
CACHE_DIR = constants.HUGGINGFACE_HUB_CACHE
USE_DEFAULT_VALUE = None  # Sentinel value for default parameter handling


# ============================================================================
# FILE AND MODEL LOADING UTILITIES
# ============================================================================

def select_compatible_kwargs(
        kwargs_dict: dict[str, Any], callable: collections.abc.Callable
) -> dict[str, Any]:
    """
    Filter kwargs to only include those compatible with a callable's signature.

    This prevents TypeError when passing kwargs that a function doesn't accept.
    Useful when chaining function calls with overlapping but not identical parameters.

    Args:
        kwargs_dict: Dictionary of keyword arguments
        callable: Function to check compatibility against

    Returns:
        Filtered dictionary containing only compatible kwargs
    """
    return {k: v for k, v in kwargs_dict.items() if k in inspect.getfullargspec(callable).args}


def download_file_from_hf(
        repo_name: str,
        file_name: str,
        subfolder: str = ".",
        cache_dir: Optional[str] = CACHE_DIR,
        force_is_torch: bool = False,
        **kwargs: Any,
):
    """
    Download and load files from Hugging Face Hub with automatic format detection.

    Automatically detects file type and loads appropriately:
    - .pth files: loaded as PyTorch tensors
    - .json files: loaded as Python dictionaries
    - Other files: returns file path

    Args:
        repo_name: Hugging Face repository name (e.g., "microsoft/DialoGPT-medium")
        file_name: Name of file to download
        subfolder: Subfolder within repository (default: root)
        cache_dir: Local cache directory for downloads
        force_is_torch: Force loading as PyTorch file even if extension doesn't match
        **kwargs: Additional arguments passed to hf_hub_download

    Returns:
        Loaded file content or file path
    """
    file_path = hf_hub_download(
        repo_id=repo_name,
        filename=file_name,
        subfolder=subfolder,
        cache_dir=cache_dir,
        **select_compatible_kwargs(kwargs, hf_hub_download),
    )
    if file_path.endswith(".pth") or force_is_torch:
        return torch.load(file_path, map_location="cpu", weights_only=False)
    elif file_path.endswith(".json"):
        return json.load(open(file_path, "r"))
    else:
        print("File type not supported:", file_path.split(".")[-1])
        return file_path


def clear_huggingface_cache():
    """Delete the entire Hugging Face cache directory to free up disk space."""
    print("Deleting Hugging Face cache directory and all its contents.")
    shutil.rmtree(CACHE_DIR)


# ============================================================================
# DEBUGGING AND TENSOR UTILITIES
# ============================================================================

def print_gpu_mem(step_name: str = ""):
    """Print current GPU memory allocation for debugging memory usage."""
    # Fixed typo: meemory -> memory, 2e30 -> 2e9 (for GiB conversion)
    print(f"{step_name} ~ {np.round(torch.cuda.memory_allocated() / 2e9, 2)} GiB allocated on GPU.")


def get_corner(tensor: Any, n: int = 3):
    """
    Get the top-left corner of a tensor for quick inspection.

    Useful for examining large tensors without printing the entire content.
    Works with both regular tensors and FactoredMatrix objects.

    Args:
        tensor: Input tensor or FactoredMatrix
        n: Size of corner to extract (n x n x ... for each dimension)

    Returns:
        Corner subset of the tensor
    """
    if isinstance(tensor, torch.Tensor):
        return tensor[tuple(slice(n) for _ in range(tensor.ndim))]
    elif isinstance(tensor, FactoredMatrix):
        return tensor[tuple(slice(n) for _ in range(tensor.ndim))].AB


def to_numpy(tensor: Any):
    """
    Convert various tensor types to numpy arrays.

    Handles multiple input types commonly used in ML workflows:
    - PyTorch tensors (detaches from computation graph)
    - NumPy arrays (passed through)
    - Lists/tuples (converted to arrays)
    - Scalars (converted to 0-dim arrays)

    Args:
        tensor: Input to convert

    Returns:
        NumPy array representation
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (torch.Tensor, torch.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")


# ============================================================================
# LANGUAGE MODEL EVALUATION METRICS
# ============================================================================

def lm_cross_entropy_loss(
        logits: Float[torch.Tensor, "batch pos d_vocab"],
        tokens: Int[torch.Tensor, "batch pos"],
        attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
        per_token: bool = False,
) -> Union[Float[torch.Tensor, ""], Float[torch.Tensor, "batch pos"]]:
    """
    Calculate cross-entropy loss for language modeling.

    Standard next-token prediction loss. Compares model predictions at each position
    with the actual next token in the sequence.

    Args:
        logits: Model output logits [batch, sequence_length, vocab_size]
        tokens: Ground truth token IDs [batch, sequence_length]
        attention_mask: Optional mask to ignore padding tokens
        per_token: If True, return loss per token; if False, return average loss

    Returns:
        Cross-entropy loss (scalar or per-token)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Get log probabilities for the actual next tokens
    predicted_log_probs = log_probs[..., :-1, :].gather(dim=-1, index=tokens[..., 1:, None])[..., 0]

    if attention_mask is not None:
        # Only count loss where both current and next tokens are not padding
        next_token_mask = torch.logical_and(attention_mask[:, :-1], attention_mask[:, 1:])
        predicted_log_probs *= next_token_mask
        n_tokens = next_token_mask.sum().item()
    else:
        n_tokens = predicted_log_probs.numel()

    if per_token:
        return -predicted_log_probs
    else:
        return -predicted_log_probs.sum() / n_tokens


def lm_accuracy(
        logits: Float[torch.Tensor, "batch pos d_vocab"],
        tokens: Int[torch.Tensor, "batch pos"],
        per_token: bool = False,
) -> Union[Float[torch.Tensor, ""], Float[torch.Tensor, "batch pos"]]:
    """
    Calculate next-token prediction accuracy.

    Measures how often the model's top prediction matches the actual next token.

    Args:
        logits: Model output logits [batch, sequence_length, vocab_size]
        tokens: Ground truth token IDs [batch, sequence_length]
        per_token: If True, return accuracy per token; if False, return overall accuracy

    Returns:
        Accuracy (scalar or per-token)
    """
    top_prediction = logits.argmax(dim=-1)
    correct_matches = top_prediction[:, :-1] == tokens[:, 1:]
    if per_token:
        return correct_matches
    else:
        return correct_matches.sum() / correct_matches.numel()


# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

def gelu_new(
        input: Float[torch.Tensor, "batch pos d_mlp"]
) -> Float[torch.Tensor, "batch pos d_mlp"]:
    """
    GELU activation function using tanh approximation.

    Gaussian Error Linear Unit - smoother alternative to ReLU.
    Uses tanh approximation for efficiency.
    """
    # Fixed typo: sart -> sqrt
    return (
            0.5
            * input
            * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    )


def gelu_fast(
        input: Float[torch.Tensor, "batch pos d_mlp"]
) -> Float[torch.Tensor, "batch pos d_mlp"]:
    """
    Fast GELU approximation with precomputed constants.

    Slightly faster than gelu_new due to precomputed coefficient.
    """
    return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


def gelu_pytorch_tanh(input: torch.Tensor) -> torch.Tensor:
    """Standard PyTorch GELU with tanh approximation."""
    return F.gelu(input, approximate="tanh")


def solu(input: Float[torch.Tensor, "batch pos d_mlp"]) -> Float[torch.Tensor, "batch pos d_mlp"]:
    """
    SoLU (Softmax Linear Unit) activation function.

    Multiplies input by its softmax, creating sparse activations.
    Used in some interpretability-focused models.
    """
    return input * F.softmax(input, dim=-1)


# Dictionary mapping activation function names to implementations
ACTIVATION_FN_DICT = {
    "solu": solu,
    "solu_ln": solu,  # SoLU with layer norm (same function)
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "silu": F.silu,  # Swish activation
    "relu": F.relu,
    "gelu": F.gelu,
    "gelu_pytorch_tanh": gelu_pytorch_tanh,
}


# ============================================================================
# WEIGHT INITIALIZATION FUNCTIONS
# ============================================================================

def calc_fan_in_and_fan_out(tensor: torch.Tensor) -> tuple[int, int]:
    """
    Calculate fan-in and fan-out for weight initialization.

    Fan-in: number of input connections to a neuron
    Fan-out: number of output connections from a neuron
    Used for Xavier/Kaiming initialization schemes.

    Args:
        tensor: Weight tensor to analyze

    Returns:
        Tuple of (fan_in, fan_out)
    """
    shape = tensor.shape
    if len(shape) == 0:
        raise ValueError("Fan in and fan out cannot be computed for scalars.")
    elif len(shape) == 1:
        fan_in = 1
        fan_out = shape[0]
    elif len(shape) == 2:  # Standard linear layer
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 3:  # Could be embedding or 1D conv
        fan_in = shape[1]
        fan_out = shape[0] * shape[2]
    else:
        raise ValueError(f"Fan in and fan out cannot be computed for shape {shape} tensors.")
    return fan_in, fan_out


def init_xavier_uniform_(param: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """
    Xavier uniform initialization.

    Initializes weights uniformly based on fan-in and fan-out.
    Good for layers with tanh or sigmoid activations.
    """
    fan_in, fan_out = calc_fan_in_and_fan_out(param)
    max_val = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return nn.init.uniform_(param, -max_val, max_val)


def init_xavier_normal_(param: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """Xavier normal initialization - Gaussian version of Xavier uniform."""
    fan_in, fan_out = calc_fan_in_and_fan_out(param)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return nn.init.normal_(param, mean=0.0, std=std)


def init_kaiming_uniform_(
        param: torch.Tensor,
        a: float = 0,  # Fixed typo: flaot -> float
        nonlinearity: str = "relu",
        gain: float = 1.0,
        mode: str = "fan_in",
) -> torch.Tensor:
    """
    Kaiming uniform initialization.

    Also known as He initialization. Good for ReLU and variants.
    Accounts for the nonlinearity to maintain activation variance.
    """
    fan_in, fan_out = calc_fan_in_and_fan_out(param)
    fan = fan_in if mode == "fan_in" else fan_out
    gain *= nn.init.calculate_gain(nonlinearity, a)
    max_val = gain * np.sqrt(3.0 / fan)
    return nn.init.uniform_(param, -max_val, max_val)


def init_kaiming_normal_(
        param: torch.Tensor,
        a: float = 0,
        nonlinearity: str = "relu",
        gain: float = 1.0,
        mode: str = "fan_in",
) -> torch.Tensor:
    """Kaiming normal initialization - Gaussian version of Kaiming uniform."""
    fan_in, fan_out = calc_fan_in_and_fan_out(param)
    fan = fan_in if mode == "fan_in" else fan_out
    gain *= nn.init.calculate_gain(nonlinearity, a)
    std = gain * np.sqrt(1.0 / fan)
    return nn.init.normal_(param, mean=0.0, std=std)


# ============================================================================
# DATASET PROCESSING UTILITIES
# ============================================================================

def keep_single_column(dataset: Dataset, col_name: str):
    """
    Remove all columns from dataset except the specified one.

    Useful for preprocessing datasets to contain only the text column
    before tokenization.
    """
    for key in dataset.features:
        if key != col_name:
            dataset = dataset.remove_columns(key)
    return dataset


def tokenize_and_concatenate(
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        streaming: bool = False,
        max_length: int = 1024,
        column_name: str = "text",
        add_bos_token: bool = True,
        num_proc: int = 10,
) -> Dataset:
    """
    Tokenize text dataset and concatenate into fixed-length sequences.

    This function processes a text dataset for language model training:
    1. Concatenates all text with EOS tokens as separators
    2. Tokenizes the concatenated text in chunks (for memory efficiency)
    3. Splits into fixed-length sequences for training
    4. Optionally adds BOS tokens at the start of each sequence

    Args:
        dataset: Hugging Face dataset with text column
        tokenizer: Tokenizer to use
        streaming: Whether dataset is streaming
        max_length: Maximum sequence length
        column_name: Name of text column in dataset
        add_bos_token: Whether to add BOS token at start of sequences
        num_proc: Number of processes for parallel tokenization

    Returns:
        Tokenized dataset with 'tokens' column
    """
    dataset = keep_single_column(dataset, column_name)

    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # Reserve space for BOS token if needed
    if add_bos_token:
        seq_len = max_length - 1
    else:
        seq_len = max_length

    def tokenize_function(examples: dict[str, list[str]]) -> dict[str, np.ndarray]:
        """Inner function to tokenize batches of text."""
        text = examples[column_name]
        assert tokenizer.eos_token is not None, "Tokenizer must have an EOS token"

        # Join all texts with EOS tokens
        full_text = tokenizer.eos_token.join(text)
        if not full_text.strip():
            return {"tokens": np.array([], dtype=np.int64)}

        # Tokenize in chunks to avoid memory issues with very long texts
        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [full_text[i * chunk_length: (i + 1) * chunk_length] for i in range(num_chunks)]
        tokens = tokenizer(chunks, return_tensors="np", padding=True)["input_ids"].flatten()

        # Remove padding tokens
        tokens = tokens[tokens != tokenizer.pad_token_id]
        num_tokens = len(tokens)

        # Handle short sequences
        if num_tokens < seq_len:
            num_batches = 1
            tokens = tokens[:seq_len]
            if len(tokens) < seq_len:
                padding_length = seq_len - len(tokens)
                padding = np.full(padding_length, tokenizer.pad_token_id)
                tokens = np.concatenate([tokens, padding], axis=0)
        else:
            # Split into multiple sequences
            num_batches = num_tokens // seq_len
            tokens = tokens[: seq_len * num_batches]

        # Reshape into sequences
        tokens = einops.rearrange(
            tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
        )

        # Add BOS tokens if requested
        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)

        return {"tokens": tokens}

    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=(num_proc if not streaming else None),
        remove_columns=[column_name],
    )
    tokenized_dataset.set_format(type="torch", columns=["tokens"])
    return tokenized_dataset


# ============================================================================
# TEXT GENERATION UTILITIES
# ============================================================================

def sample_logits(
        final_logits: Float[torch.Tensor, "batch d_vocab"],
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        tokens: Optional[Int[torch.Tensor, "batch pos"]] = None,
) -> Int[torch.Tensor, "batch"]:
    """
    Sample tokens from logits with various sampling strategies.

    Supports multiple sampling methods:
    - Temperature scaling: Higher values = more random
    - Top-k sampling: Only consider k most likely tokens
    - Top-p (nucleus) sampling: Consider tokens until cumulative prob reaches p
    - Frequency penalty: Reduce probability of recently seen tokens

    Args:
        final_logits: Model output logits [batch, vocab_size]
        top_k: Number of top tokens to consider (None = consider all)
        top_p: Cumulative probability threshold for nucleus sampling
        temperature: Sampling temperature (0 = greedy, >1 = more random)
        freq_penalty: Penalty for frequently occurring tokens
        tokens: Previous tokens for frequency penalty calculation

    Returns:
        Sampled token indices [batch]
    """
    # Example debug code (should be removed in production)
    logits = torch.randn(4)
    print(logits)
    np.unique(np.array([sample_logits(logits, top_k=2).item() for i in range(1000)]), return_counts=True)

    # Greedy sampling if temperature is 0
    if temperature == 0.0:
        return final_logits.argmax(dim=-1)
    else:
        # Apply temperature scaling
        final_logits = final_logits / temperature

        # Apply frequency penalty if specified
        if freq_penalty > 0:
            assert tokens is not None, "Must provide input_tokens if applying a frequency penalty"
            assert (
                    len(tokens.shape) == 2
            ), "Frequency penalty do not support input in the form of embeddings"

            for batch_index in range(final_logits.shape[0]):
                # Reduce logits for frequently occurring tokens
                final_logits[batch_index] = final_logits[batch_index] - freq_penalty * torch.bincount(
                    tokens[batch_index], minlength=final_logits.shape[-1])

        # Apply top-k filtering
        if top_k is not None:
            assert top_k > 0, "top_k has to be greater than 0"
            top_logits, _ = final_logits.topk(top_k, dim=-1)
            indices_to_remove = final_logits < top_logits[..., -1].unsqueeze(-1)
            final_logits = final_logits.masked_fill(indices_to_remove, -float("inf"))

        # Apply top-p (nucleus) filtering
        elif top_p is not None:  # Fixed typo: "top p" -> "top_p"
            assert 1.0 >= top_p > 0.0, "top_p has to be in (0,1)"
            sorted_logits, sorted_indices = torch.sort(final_logits, descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Always keep at least the top token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Map back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            final_logits = final_logits.masked_fill(indices_to_remove, -float("inf"))

        # Sample from the filtered distribution
        final_logits = final_logits.to(torch.float32)
        return torch.distributions.categorical.Categorical(logits=final_logits).sample()


# ============================================================================
# SLICING AND INDEXING UTILITIES
# ============================================================================

# Type definitions for flexible slicing input
SliceInput = Optional[
    Union[
        int,
        Tuple[int,],
        Tuple[int, int],
        Tuple[int, int, int],
        List[int],
        torch.Tensor,
        np.ndarray,
    ]
]


class Slice:
    """
    Flexible slicing utility for tensors.

    Supports multiple input formats:
    - Integers: single index
    - Tuples: slice(start, stop, step)
    - Lists/arrays: advanced indexing
    - None: identity slice (all elements)

    Useful for activation patching and analysis where you need
    to specify which positions/layers/heads to analyze.
    """
    slice: Union[int, slice, np.ndarray]

    def __init__(
            self,
            input_slice: SliceInput = None,
    ):
        """Initialize slice from various input formats."""
        if isinstance(input_slice, tuple):
            self.slice = slice(*input_slice)
            self.mode = "slice"
        elif isinstance(input_slice, int):
            self.slice = input_slice
            self.mode = "int"
        elif isinstance(input_slice, slice):
            self.slice = input_slice
            self.mode = "slice"
        elif type(input_slice) in [list, torch.Tensor, np.ndarray]:
            self.slice = to_numpy(input_slice)
            self.mode = "array"
        elif input_slice is None:
            self.slice = slice(None)  # Select all
            self.mode = "identity"
        else:
            raise ValueError(f"Invalid input_slice {input_slice}")

    def apply(
            self,
            tensor: torch.Tensor,
            dim: int = 0,
    ) -> torch.Tensor:
        """
        Apply this slice to a tensor along the specified dimension.

        Args:
            tensor: Input tensor to slice
            dim: Dimension to slice along

        Returns:
            Sliced tensor
        """
        ndim = tensor.ndim
        slices = [slice(None)] * ndim  # Identity slices for all dims
        slices[dim] = self.slice  # Replace target dim with our slice
        return tensor[tuple(slices)]

    def indices(
            self,
            max_ctx: Optional[int] = None,
    ) -> Union[np.ndarray, np.int32, np.int64]:
        """
        Get the actual indices this slice represents.

        Args:
            max_ctx: Maximum context length (required for slice objects)

        Returns:
            Array of indices
        """
        if self.mode == "int":
            return np.array([self.slice], dtype=np.int64)
        if max_ctx is None:
            raise ValueError("max_ctx must be specified if slice is not an integer")
        return np.arange(max_ctx, dtype=np.int64)[self.slice]

    def __repr__(
            self,
    ) -> str:
        return f"Slice: {self.slice} Mode: {self.mode}"

    @classmethod
    def unwrap(
            cls,
            slice_input: Union["Slice", SliceInput]
    ) -> "Slice":
        """
        Convert various slice inputs to Slice objects.

        Convenience method that ensures we have a Slice object regardless
        of input format. Automatically wraps single integers in lists.
        """
        if not isinstance(slice_input, Slice):
            if isinstance(slice_input, int):
                slice_input = [slice_input]  # Wrap single ints in list
            slice_input = Slice(slice_input)
        return slice_input


# ============================================================================
# ACTIVATION NAME UTILITIES
# ============================================================================

def get_act_name(
        name: str,
        layer: Optional[Union[int, str]] = None,
        layer_type: Optional[str] = None
):
    """
    Convert activation names to standard TransformerLens hook names.

    TransformerLens uses a specific naming convention for hooks:
    - "blocks.{layer}.{component}.hook_{activation}"

    This function handles various input formats and aliases to generate
    the correct hook names for activation patching.

    Args:
        name: Base activation name (e.g., "q", "k", "v", "pattern", "resid_pre")
        layer: Layer number or already formatted layer string
        layer_type: Component type ("attn", "mlp", or aliases)

    Returns:
        Properly formatted activation hook name

    Examples:
        get_act_name("q", 5) -> "blocks.5.attn.hook_q"
        get_act_name("resid_pre", 0) -> "blocks.0.hook_resid_pre"
        get_act_name("pattern", 3) -> "blocks.3.attn.hook_pattern"
    """
    # If already a full hook name, return as-is
    if ("." in name or name.startswith("hook_")) and layer is None and layer_type is None:
        return name

    # Try to parse combined name like "attn5" or "mlp3b"
    match = re.match(r"([a-z]+)(\d+)([a-z]?.*)", name)
    if match is not None:
        name, layer, layer_type = match.groups(0)

    # Layer type aliases for convenience
    layer_type_alias = {
        "a": "attn",
        "m": "mlp",
        "b": "",  # Block-level (no specific component)
        "block": "",
        "blocks": "",
        "attention": "attn",
    }

    # Activation name aliases
    act_name_alias = {
        "attn": "pattern",  # "attn" usually means attention patterns
        "attn_logits": "attn_scores",  # Pre-softmax attention scores
        "key": "k",
        "query": "q",
        "value": "v",
        "mlp_pre": "pre",  # Pre-activation MLP
        "mlp_mid": "mid",  # Mid-activation MLP
        "mlp_post": "post",  # Post-activation MLP
    }

    # Layer norm related names
    layer_norm_names = ["scale", "normalized"]

    # Apply name aliases
    if name in act_name_alias:
        name = act_name_alias[name]

    # Build the full activation name
    full_act_name = ""

    # Add layer specification
    if layer is not None:
        full_act_name += f"blocks.{layer}."

    # Infer layer type from activation name if not specified
    if name in [
        "k", "v", "q", "z",  # Attention vectors
        "rot_k",  # Rotary position encoding applied to keys
        "result",  # Attention output
        "pattern",  # Attention patterns
        "attn_scores"  # Pre-softmax attention scores
    ]:
        layer_type = "attn"
    elif name in ["pre", "post", "mid", "pre_linear"]:  # MLP components
        layer_type = "mlp"
    elif layer_type in layer_type_alias:
        layer_type = layer_type_alias[layer_type]

    # Add layer type if specified
    if layer_type:
        full_act_name += f"{layer_type}."

    # Add the hook prefix and activation name
    full_act_name += f"hook_{name}"

    # Special case for layer norm when no layer is specified (final layer norm)
    if name in layer_norm_names and layer is None:
        full_act_name = f"ln_final.{full_act_name}"

    return full_act_name


# ============================================================================
# TENSOR MANIPULATION UTILITIES
# ============================================================================

def remove_batch_dim(tensor: Float[torch.Tensor, "1 ..."]) -> Float[torch.Tensor, "..."]:
    """
    Remove batch dimension if batch size is 1.

    Convenience function for converting single-item batches to unbatched tensors.
    Useful when working with individual examples.
    """
    if tensor.shape[0] == 1:
        return tensor.squeeze(0)
    else:
        return tensor


def transpose(tensor: Float[torch.Tensor, "...a b"]) -> Float[torch.Tensor, "...b a"]:
    """Transpose the last two dimensions of a tensor."""
    return tensor.transpose(-1, -2)


# ============================================================================
# MODEL TESTING AND EVALUATION UTILITIES
# ============================================================================

def test_prompt(
        prompt: str,
        answer: Union[str, list[str]],
        model,
        prepend_space_to_answer: bool = True,
        print_details: bool = True,
        prepend_bos: Optional[bool] = USE_DEFAULT_VALUE,
        top_k: int = 10,
) -> None:
    """
    Test model performance on a specific prompt-answer pair.

    This function evaluates how well a model predicts specific answers to prompts.
    It's useful for testing model capabilities on particular tasks or checking
    if interventions (like activation patching) affect specific behaviors.

    Args:
        prompt: Input text prompt
        answer: Expected answer(s) - can be single string or list of alternatives
        model: TransformerLens model to test
        prepend_space_to_answer: Add space before answer if not present
        print_details: Whether to print detailed analysis
        prepend_bos: Whether to prepend BOS token to prompt
        top_k: Number of top predictions to show

    Prints detailed analysis including:
        - Token-level predictions and ranks
        - Logits and probabilities for target tokens
        - Top alternative predictions
    """
    # Handle single answer vs multiple alternatives
    answers = [answer] if isinstance(answer, str) else answer
    n_answers = len(answers)
    using_multiple_answers = n_answers > 1

    # Add leading space to answers if requested (common for tokenization)
    if prepend_space_to_answer:
        answers = [answer if answer.startswith(" ") else " " + answer for answer in answers]

    # Tokenize prompt and answers
    prompt_tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    answer_tokens = model.to_tokens(answers, prepend_bos=False)

    # For multiple answers, only look at first token of each
    if using_multiple_answers:
        answer_tokens = answer_tokens[:, :1]

    # Create full sequences (prompt + answer)
    prompt_tokens = prompt_tokens.repeat(answer_tokens.shape[0], 1)
    tokens = torch.cat((prompt_tokens, answer_tokens), dim=1)

    # Get string representations for display
    prompt_str_tokens = model.to_str_tokens(prompt, prepend_bos=prepend_bos)
    answer_str_tokens_list = [model.to_str_tokens(answer, prepend_bos=False) for answer in answers]

    prompt_length = len(prompt_str_tokens)
    answer_length = 1 if using_multiple_answers else len(answer_str_tokens_list[0])

    if print_details:
        print("Tokenized prompt:", prompt_str_tokens)
        if using_multiple_answers:
            print("Tokenized answers:", answer_str_tokens_list)
        else:
            print("Tokenized answer:", answer_str_tokens_list[0])

    # Run model and get predictions
    logits = model(tokens)
    probs = logits.softmax(dim=-1)

    # Analyze each answer token position
    answer_ranks = []
    for index in range(prompt_length, prompt_length + answer_length):
        answer_tokens = tokens[:, index]
        answer_str_tokens = [a[index - prompt_length] for a in answer_str_tokens_list]
        token_probs = probs[:, index - 1]  # Predictions for current position

        # Get ranking of target tokens
        sorted_token_probs, sorted_token_positions = token_probs.sort(descending=True)
        # Fixed typo: argstort -> argsort
        answer_token_ranks = sorted_token_positions.argsort(-1)[
            range(n_answers), answer_tokens.cpu()
        ].tolist()

        answer_ranks.append(
            [
                (answer_str_token, answer_token_rank)
                for answer_str_token, answer_token_rank in zip(answer_str_tokens, answer_token_ranks)
            ]
        )

        if print_details:
            # Print performance for target tokens
            rprint(
                f"Performance on answer token{'s' if n_answers > 1 else ''}:\n"
                + "\n".join(
                    [
                        f"[b]Rank: {answer_token_ranks[i]: <8} Logit: {logits[i, index - 1, answer_tokens[i]].item():5.2f} Prob: {token_probs[i, answer_tokens[i]].item():6.2%} Token: |{answer_str_tokens[i]}|[/b]"
                        for i in range(n_answers)
                    ]
                )
            )
            # Print top predictions for comparison
            for i in range(top_k):
                print(
                    f"Top {i}th token. Logit: {logits[0, index - 1, sorted_token_positions[0, i]].item():5.2f} Prob: {sorted_token_probs[0, i].item():6.2%} Token: |{model.to_string(sorted_token_positions[0, i])}|"
                )

    # Print final summary
    if not using_multiple_answers:
        single_answer_ranks = [r[0] for r in answer_ranks]
        rprint(f"[b]Ranks of the answer tokens: [/b] {single_answer_ranks}")
    else:
        rprint(f"[b]Ranks of the answer tokens: [/b] {answer_ranks}")


# ============================================================================
# MATRIX COMPOSITION ANALYSIS
# ============================================================================

def composition_scores(
        left: "FactoredMatrix", right: "FactoredMatrix", broadcast_dims=True
) -> Union[
    Float[torch.Tensor, "leading_dims"], Float[torch.Tensor, "leading_dims_left_and_right"]
]:
    """
    Calculate composition scores between two factored matrices.

    Measures how much two matrices compose together by comparing the norm
    of their composition to the product of their individual norms.
    Higher scores indicate stronger composition.

    Used in mechanistic interpretability to understand how different
    components (attention heads, MLPs) interact with each other.

    Args:
        left: Left matrix in composition (e.g., OV matrix of attention head)
        right: Right matrix in composition (e.g., QK matrix of attention head)
        broadcast_dims: Whether to broadcast dimensions for batch operations

    Returns:
        Composition scores normalized by individual matrix norms
    """
    if broadcast_dims:
        r_leading = right.ndim - 2
        l_leading = left.ndim - 2  # Fixed typo: lef -> left
        # Add dimensions for broadcasting
        for i in range(l_leading):
            right = right.unsqueeze(i)
        for i in range(r_leading):
            left = left.unsqueeze(i + l_leading)

    assert (
            left.rdim == right.ldim
    ), f"Composition scores require left.rdim==right.ldim, shapes were left: {left.shape}, right:{right.shape}"

    # Collapse to 2D matrices for computation
    new_right = right.collapse_r()
    new_left = left.collapse_l()

    # Calculate norms
    r_norms = new_right.norm(dim=[-2, -1])
    l_norms = new_left.norm(dim=[-2, -1])
    comp_norms = (new_left @ new_right).norm(dim=[-2, -1])

    # Return normalized composition strength
    return comp_norms / r_norms / l_norms


# ============================================================================
# DATASET LOADING UTILITIES
# ============================================================================

def get_dataset(dataset_name: str, **kwargs) -> Dataset:
    """
    Load common datasets with convenient aliases.

    Provides easy access to frequently used datasets in language modeling
    and mechanistic interpretability research.

    Args:
        dataset_name: Name or alias of dataset to load
        **kwargs: Additional arguments passed to load_dataset

    Returns:
        Loaded Hugging Face dataset

    Supported datasets:
        - openwebtext/owt: Web text dataset
        - pile: The Pile dataset
        - c4: C4 (Colossal Clean Crawled Corpus)
        - code/python: Code dataset
        - c4_code/c4-code: Mixed C4 and code
        - wiki: Wikipedia dataset
    """
    dataset_aliases = {
        "openwebtext": "stas/openwebtext-10k",
        "owt": "stas/openwebtext-10k",
        "pile": "NeelNanda/pile-10k",
        "c4": "NeelNanda/c4-10k",
        "code": "NeelNanda/code-10k",
        "python": "NeelNanda/code-10k",
        "c4_code": "NeelNanda/c4-code-20k",
        "c4-code": "NeelNanda/c4-code-20k",
        "wiki": "NeelNanda/wiki-10k"
    }

    if dataset_name in dataset_aliases:
        dataset = load_dataset(dataset_aliases[dataset_name], split="train", **kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return dataset


# ============================================================================
# MATRIX ANALYSIS UTILITIES
# ============================================================================

def is_square(x: torch.Tensor) -> bool:
    """Check if tensor is a square matrix."""
    return x.ndim == 2 and x.shape[0] == x.shape[1]


def is_lower_triangular(x: torch.Tensor) -> bool:
    """Check if matrix is lower triangular."""
    if not is_square(x):
        return False
    return x.equal(x.tril())


def check_structure(t1: torch.Tensor, t2: torch.Tensor, *, verbose: bool = False) -> None:
    """
    Compare the ordering structure of two matrices.

    Checks if two matrices have the same relative ordering along rows and columns.
    Useful for verifying that different implementations produce structurally
    similar results (e.g., attention patterns).

    Args:
        t1: First matrix to compare
        t2: Second matrix to compare
        verbose: Whether to print detailed mismatch information

    Prints:
        - "PASSED" if structures match
        - Lists of mismatched rows/columns if structures differ
    """
    assert t1.ndim == 2
    assert t1.shape == t2.shape
    n_rows, n_cols = cast(Tuple[int, int], t1.shape)

    # Check row-wise ordering
    if verbose:
        print("Checking rows")
    row_mismatch = []
    for row_i in range(n_rows - 1):
        # Compare adjacent rows element-wise
        t1_result = t1[row_i].ge(t1[row_i + 1])
        t2_result = t2[row_i].ge(t2[row_i + 1])
        if any(t1_result != t2_result):
            row_mismatch.append(row_i)
            if verbose:
                print(f"\trows {row_i}:{row_i + 1}")
                print(f"\tt1: {t1_result.tolist()}")
                print(f"\tt2: {t2_result.tolist()}")

    # Check column-wise ordering
    if verbose:
        print("Checking columns")
    col_mismatch = []
    for col_i in range(n_cols - 1):
        # Compare adjacent columns element-wise
        t1_result = t1[:, col_i].ge(t1[:, col_i + 1])
        t2_result = t2[:, col_i].ge(t2[:, col_i + 1])
        if any(t1_result != t2_result):
            col_mismatch.append(col_i)
            if verbose:
                print(f"\tcols {col_i}:{col_i + 1}")
                print(f"\tt1: {t1_result.tolist()}")
                print(f"\tt2: {t2_result.tolist()}")

    # Print results
    if not row_mismatch and not col_mismatch:
        print("PASSED")
    elif row_mismatch:
        print(f"row mismatch: {row_mismatch}")
    elif col_mismatch:
        print(f"column mismatch: {col_mismatch}")


# ============================================================================
# DEVICE AND HARDWARE UTILITIES
# ============================================================================

def get_device():
    """
    Get the best available device for computation.

    Priority order:
    1. CUDA (NVIDIA GPU) if available
    2. MPS (Apple Silicon GPU) if available and PyTorch >= 2.0
    3. CPU as fallback

    Returns:
        torch.device object for the best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        major_version = int(torch.__version__.split(".")[0])
        if major_version >= 2:  # MPS support improved significantly in PyTorch 2.0
            return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
# PARAMETER OVERRIDE UTILITIES
# ============================================================================

def override_or_use_default_value(
        default_flag: Any,
        override: Optional[Any] = None,
) -> Any:
    """
    Use override value if provided, otherwise use default.

    Simple utility for parameter handling with optional overrides.
    """
    return override if override is not None else default_flag


# ============================================================================
# ATTENTION AND POSITIONAL UTILITIES
# ============================================================================

def get_offset_position_ids(
        past_kv_pos_offset: int,
        attention_mask: Int[torch.Tensor, "batch offset_pos"],
) -> Int[torch.Tensor, "batch pos"]:
    """
    Generate position IDs accounting for past key-value cache offset.

    Used in models with KV caching where position IDs need to account
    for previously cached tokens.

    Args:
        past_kv_pos_offset: Number of tokens in past KV cache
        attention_mask: Mask indicating valid positions

    Returns:
        Position IDs starting from the offset
    """
    # Calculate cumulative position based on attention mask
    shifted_position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids = shifted_position_ids.masked_fill(shifted_position_ids < 0, 0)
    return position_ids[:, past_kv_pos_offset:]


def get_cumsum_along_dim(tensor, dim, reverse=False):
    """
    Calculate cumulative sum along a dimension, optionally in reverse.

    Args:
        tensor: Input tensor
        dim: Dimension along which to calculate cumsum
        reverse: If True, calculate cumsum from end to start

    Returns:
        Tensor with cumulative sum along specified dimension
    """
    if reverse:
        tensor = tensor.flip(dims=(dim,))
    cumsum = tensor.cumsum(dim=dim)
    if reverse:
        cumsum = cumsum.flip(dims=(dim,))
    return cumsum


def get_attention_mask(
        tokenizer: transformers.PreTrainedTokenizerBase,
        tokens: torch.Tensor,
        prepend_bos: bool,
) -> torch.Tensor:
    """
    Generate attention mask for tokenized input.

    Creates mask that ignores padding tokens and handles different
    padding strategies (left vs right padding).

    Args:
        tokenizer: Tokenizer used to create tokens
        tokens: Token tensor [batch, seq_len]
        prepend_bos: Whether BOS token was prepended

    Returns:
        Attention mask (1 for valid tokens, 0 for padding)
    """
    attention_mask = torch.ones_like(tokens)

    if tokenizer is None:
        return attention_mask

    # Identify non-padding tokens
    is_not_pad_token = tokens.ne(tokenizer.pad_token_id)

    # Fixed typo: padding_sie -> padding_side
    if tokenizer.padding_side == "right":
        # Right padding: mask trailing padding tokens
        is_trailing_pad = get_cumsum_along_dim(is_not_pad_token, -1, reverse=True) == 0
        attention_mask[is_trailing_pad] = 0
    else:
        # Left padding: mask leading padding tokens
        is_leading_pad = get_cumsum_along_dim(is_not_pad_token, -1, reverse=False) == 0
        attention_mask[is_leading_pad] = 0

        # Special case: if BOS token ID equals pad token ID, unmask the BOS position
        if prepend_bos and tokenizer.bos_token_id == tokenizer.pad_token_id:
            pad_bos_positions = is_leading_pad.sum(-1) - 1
            attention_mask[torch.arange(attention_mask.shape[0]), pad_bos_positions] = 1

    return attention_mask


def repeat_along_head_dimension(
        tensor: Float[torch.Tensor, "batch pos d_model"],
        n_heads: int,
        clone_tensor=True,
):
    """
    Repeat tensor along head dimension for multi-head operations.

    Useful when you need to broadcast a tensor across attention heads,
    such as position encodings or bias terms.

    Args:
        tensor: Input tensor [batch, pos, d_model]
        n_heads: Number of attention heads
        clone_tensor: Whether to clone the result (prevents in-place modifications)

    Returns:
        Repeated tensor [batch, pos, n_heads, d_model]
    """
    repeated_tensor = einops.repeat(
        tensor,
        "batch pos d_model -> batch pos n_heads d_model",
        n_heads=n_heads,
    )
    if clone_tensor:
        return repeated_tensor.clone()
    else:
        return repeated_tensor


# ============================================================================
# ATTRIBUTE ACCESS UTILITIES
# ============================================================================

def get_nested_attr(obj, attr_str):
    """
    Get nested attribute using dot notation.

    Example: get_nested_attr(model, "transformer.wte.weight")
    """
    attrs = attr_str.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def set_nested_attr(obj, attr_str, value):
    """
    Set nested attribute using dot notation.

    Example: set_nested_attr(model, "transformer.wte.weight", new_weight)
    """
    attrs = attr_str.split(".")
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)


# ============================================================================
# CONTEXT MANAGERS FOR TEMPORARY OVERRIDES
# ============================================================================

class LocallyOverridenDefaults:
    """
    Context manager for temporarily overriding model defaults.

    Allows you to temporarily change model configuration (like prepend_bos
    or padding_side) within a context, then automatically restore the
    original values when exiting.

    Usage:
        with LocallyOverridenDefaults(model, prepend_bos=True):
            # model.cfg.default_prepend_bos is temporarily True
            tokens = model.to_tokens(text)
        # model.cfg.default_prepend_bos is restored to original value
    """

    def __init__(self, model, **overrides):
        """
        Initialize with model and parameter overrides.

        Args:
            model: Model whose defaults to override
            **overrides: Parameters to override (e.g., prepend_bos=True)
        """
        self.model = model
        self.overrides = overrides

        # Define valid parameters and their locations
        self.values_with_defaults = {
            "prepend_bos": {
                "default_location": "model.cfg.default_prepend_bos",
                "valid_values": [USE_DEFAULT_VALUE, True, False],
                "skip_overriding": False,
                "default_value_to_restore": None,
            },
            "padding_side": {
                "default_location": "model.tokenizer.padding_side",
                "valid_values": [USE_DEFAULT_VALUE, "left", "right"],
                "skip_overriding": model.tokenizer is None,
                "default_value_to_restore": None,
            }
        }

        # Validate override parameters
        for override in overrides:
            assert override in self.values_with_defaults, (
                f"{override} is not a valid parameter to override."
                f"Valid parameters are {self.values_with_defaults.keys()}."
            )

    def __enter__(self):
        """Apply overrides when entering context."""
        for property, override in self.overrides.items():
            info = self.values_with_defaults[property]
            if info["skip_overriding"]:
                continue

            # Validate override value
            valid_values = info["valid_values"]
            assert (
                    override in valid_values
            ), f"{property} must be one of {valid_values}, but got {override}."

            # Store original value and apply override
            default_location = info["default_location"]
            default_value = get_nested_attr(self, default_location)
            info["default_value_to_restore"] = deepcopy(default_value)

            locally_overriden_value = override_or_use_default_value(default_value, override)
            set_nested_attr(self, default_location, locally_overriden_value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original values when exiting context."""
        for property in self.overrides:
            info = self.values_with_defaults[property]
            if info["skip_overriding"]:
                continue

            # Restore original value
            default_location = info["default_location"]
            default_value = info["default_value_to_restore"]
            set_nested_attr(self, default_location, default_value)


# ============================================================================
# TOKENIZER UTILITIES FOR BOS TOKEN HANDLING
# ============================================================================

def get_tokenizer_with_bos(
        tokenizer: transformers.PreTrainedTokenizerBase,
) -> transformers.PreTrainedTokenizerBase:
    """
    Get a version of the tokenizer that includes BOS tokens.

    If the tokenizer already adds BOS tokens, returns it unchanged.
    Otherwise, creates a new tokenizer instance with add_bos_token=True.

    Args:
        tokenizer: Original tokenizer

    Returns:
        Tokenizer that adds BOS tokens
    """
    init_kwargs = deepcopy(tokenizer.init_kwargs)
    pretrained_model_name_or_path = init_kwargs.pop("name_or_path")
    add_bos_token = init_kwargs.pop("add_bos_token", None)

    if add_bos_token is None:
        add_bos_token = getattr(tokenizer, "add_bos_token", False)

    if add_bos_token:
        tokenizer_with_bos = tokenizer
    else:
        # Create new tokenizer with BOS token enabled
        huggingface_token = os.environ.get("HF_TOKEN", "")
        tokenizer_with_bos = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            add_bos_token=True,
            token=huggingface_token if len(huggingface_token) > 0 else None,
            **init_kwargs,
        )
    return tokenizer_with_bos


def get_input_with_manually_prepended_bos(
        tokenizer: transformers.PreTrainedTokenizerBase, input: Union[str, list[str]]
):
    """
    Manually prepend BOS token to input text(s).

    Useful when you need BOS tokens but don't want to modify the tokenizer.

    Args:
        tokenizer: Tokenizer (used to get BOS token)
        input: Text string or list of strings

    Returns:
        Input with BOS token prepended
    """
    if isinstance(input, str):
        input = tokenizer.bos_token + input
    else:
        input = [tokenizer.bos_token + string for string in input]
    return input


def get_tokens_with_bos_removed(
        tokenizer: transformers.PreTrainedTokenizerBase,
        tokens: Int[torch.Tensor, "batch pos"],
):
    """
    Remove BOS tokens from tokenized input.

    Handles both left and right padding, and cases where BOS token
    ID equals padding token ID.

    Args:
        tokenizer: Tokenizer used to create tokens
        tokens: Token tensor with BOS tokens

    Returns:
        Token tensor with BOS tokens removed
    """
    if tokenizer.padding_side == "right":
        # Right padding: BOS is always first token
        return tokens[..., 1:]
    else:
        # Left padding: need to find actual BOS position
        bos_removed_shape = list(tokens.shape)
        bos_removed_shape[-1] -= 1

        if tokenizer.bos_token_id == tokenizer.pad_token_id:
            # BOS and padding share same token ID - find real BOS position
            is_not_pad_token = tokens.ne(tokenizer.pad_token_id)
            is_leading_pad = get_cumsum_along_dim(is_not_pad_token, -1, reverse=False) == 0
            real_bos_positions = is_leading_pad.sum(-1) - 1
        else:
            # Find BOS token position directly
            real_bos_positions = (tokens == tokenizer.bos_token_id).int().argmax(-1)

        # Mark BOS positions for removal
        tokens = tokens.scatter(dim=1, index=real_bos_positions.unsqueeze(-1), value=-100)
        return tokens[tokens != -100].view(*bos_removed_shape)
    try:
        import pytest
        pytest.mark.skip(test_prompt)
    except ModuleNotFoundError:
        pass

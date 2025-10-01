from __future__ import annotations
import itertools
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union, overload
import einops
import pandas as pd
import torch
from jaxtyping import Float, Int
from torch import FloatTensor
from tqdm.auto import tqdm
from typing_extensions import Literal
import utils as utils
from activation_cache import ActivationCache
from hooked_transformer import HookedTransformer

# Type aliases for clarity
Logits = torch.Tensor
AxisNames = Literal["layer", "pos", "head_index", "head", "src_pos", "dest_pos"]

from typing import Sequence


def make_df_from_ranges(
        column_max_ranges: Sequence[int], column_names: Sequence[str]
) -> pd.DataFrame:
    """
    Create a DataFrame containing all combinations of indices for activation patching.

    This function generates a DataFrame where each row represents a unique combination
    of indices across multiple dimensions (e.g., layer, position, head). This is used
    to systematically patch activations at every possible location.

    Args:
        column_max_ranges: Maximum values for each dimension (e.g., [n_layers, seq_len, n_heads])
        column_names: Names for each dimension (e.g., ["layer", "pos", "head"])

    Returns:
        DataFrame with columns for each dimension and rows for each index combination
    """
    # Generate all possible combinations using Cartesian product
    rows = list(itertools.product(*[range(axis_max_range) for axis_max_range in column_max_ranges]))
    df = pd.DataFrame(rows, columns=column_names)
    return df


# Type aliases for activation tensors
CorruptedActivation = torch.Tensor  # Activations from corrupted/counterfactual input
PatchedActivation = torch.Tensor  # Activations after patching with clean values


# Overloaded function signatures to handle optional return of index DataFrame
@overload
def generic_activation_patch(
        model: HookedTransformer,
        corrupted_tokens: Int[torch.Tensor, "batch pos"],
        clean_cache: ActivationCache,
        patching_metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], Float[torch.Tensor, ""]],
        patch_setter: Callable[
            [CorruptedActivation, Sequence[int], ActivationCache], PatchedActivation
        ],
        activation_name: str,
        index_axis_names: Optional[Sequence[AxisNames]],
        index_df=Optional[pd.DataFrame],
        return_index_df: Literal[False] = False,
) -> torch.Tensor:
    ...


@overload
def generic_activation_patch(
        model: HookedTransformer,
        corrupted_tokens: Int[torch.Tensor, "batch pos"],
        clean_cache: ActivationCache,
        patching_metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], Float[torch.Tensor, ""]],
        patch_setter: Callable[
            [CorruptedActivation, Sequence[int], ActivationCache], PatchedActivation
        ],
        activation_name: str,
        index_axis_names: Optional[Sequence[AxisNames]],
        index_df=Optional[pd.DataFrame],
        return_index_df: Literal[True] = True,
) -> Tuple[torch.Tensor, pd.DataFrame]:
    ...


def generic_activation_patch(
        model: HookedTransformer,
        corrupted_tokens: Int[torch.Tensor, "batch pos"],
        clean_cache: ActivationCache,
        patching_metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], Float[torch.Tensor, ""]],
        patch_setter: Callable[
            [CorruptedActivation, Sequence[int], ActivationCache], PatchedActivation
        ],
        activation_name: str,
        index_axis_names: Optional[Sequence[AxisNames]] = None,
        index_df: Optional[pd.DataFrame] = None,
        return_index_df: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, pd.DataFrame]]:
    """
    Generic function for activation patching experiments in mechanistic interpretability.

    This function implements the core activation patching methodology:
    1. Run model on corrupted input
    2. At each specified location, replace corrupted activations with clean activations
    3. Measure how this affects model output using a patching metric
    4. Return results showing which activations matter most for the behavior

    Args:
        model: The transformer model to analyze
        corrupted_tokens: Input tokens that produce incorrect/undesired behavior
        clean_cache: Cached activations from clean input (correct behavior)
        patching_metric: Function that measures model performance (e.g., logit difference)
        patch_setter: Function that specifies how to patch activations at given indices
        activation_name: Name of activation to patch (e.g., "resid_pre", "z", "pattern")
        index_axis_names: Names of dimensions to iterate over (e.g., ["layer", "pos"])
        index_df: Pre-computed DataFrame of indices (alternative to index_axis_names)
        return_index_df: Whether to return the index DataFrame along with results

    Returns:
        Tensor of patching results, optionally with index DataFrame
    """

    # Set up indexing: either create new DataFrame or use provided one
    if index_df is None:
        assert index_axis_names is not None
        # Define maximum ranges for each axis type
        max_axis_range = {
            "layer": model.cfg.n_layers,
            "pos": corrupted_tokens.shape[-1],  # sequence length
            "head_index": model.cfg.n_heads,
        }
        # Add aliases for convenience
        max_axis_range["src_pos"] = max_axis_range["pos"]
        max_axis_range["dest_pos"] = max_axis_range["pos"]
        max_axis_range["head"] = max_axis_range["head_index"]

        # Get max ranges for the specified axes
        index_axis_max_range = [max_axis_range[axis_name] for axis_name in index_axis_names]
        # Create DataFrame with all index combinations
        index_df = make_df_from_ranges(index_axis_max_range, index_axis_names)
        flattened_output = False
    else:
        # Use provided DataFrame
        assert index_axis_names is None
        index_axis_max_range = index_df.max().to_list()
        flattened_output = True

    # Initialize output tensor to store patching results
    if flattened_output:
        patched_metric_output = torch.zeros(len(index_df), device=model.cfg.device)
    else:
        patched_metric_output = torch.zeros(index_axis_max_range, device=model.cfg.device)

    def patching_hook(corrupted_activation, hook, index, clean_activation):
        """Hook function that patches activations during forward pass."""
        return patch_setter(corrupted_activation, index, clean_activation)

    # Iterate through all index combinations
    for c, index_row in enumerate(tqdm((list(index_df.iterrows())))):
        index = index_row[1].to_list()  # Convert to list of indices

        # Get the specific activation name for this layer
        current_activation_name = utils.get_act_name(activation_name, layer=index[0])

        # Create hook with current index and clean activation
        current_hook = partial(
            patching_hook,
            index=index,
            clean_activation=clean_cache[current_activation_name],
        )

        # Run model with the patching hook
        patched_logits = model.run_with_hooks(
            corrupted_tokens, fwd_hooks=[(current_activation_name, current_hook)]
        )

        # Store the metric result
        if flattened_output:
            patched_metric_output[c] = patching_metric(patched_logits).item()
        else:
            patched_metric_output[tuple(index)] = patching_metric(patched_logits).item()

    if return_index_df:
        return patched_metric_output, index_df
    else:
        return patched_metric_output


# ============================================================================
# PATCH SETTER FUNCTIONS
# These functions define HOW to patch activations at specific indices
# ============================================================================

def layer_pos_patch_setter(corrupted_activation, index, clean_activation):
    """
    Patch activations at a specific layer and position.
    Used for residual stream activations, attention outputs, MLP outputs.

    Args:
        corrupted_activation: Current activation tensor
        index: [layer, position] indices
        clean_activation: Clean activation to patch in
    """
    assert len(index) == 2
    layer, pos = index
    # Replace activation at this position with clean version
    corrupted_activation[:, pos, ...] = clean_activation[:, pos, ...]
    return corrupted_activation


def layer_pos_head_vector_patch_setter(
        corrupted_activation,
        index,
        clean_activation,
):
    """
    Patch activations for a specific head at a specific position.
    Used for Q, K, V vectors and attention head outputs (z).

    Args:
        corrupted_activation: Current activation tensor [batch, pos, head, ...]
        index: [layer, position, head] indices
        clean_activation: Clean activation to patch in
    """
    assert len(index) == 3
    layer, pos, head_index = index
    # Replace activation for this head at this position
    corrupted_activation[:, pos, head_index] = clean_activation[:, pos, head_index]
    return corrupted_activation


def layer_head_vector_patch_setter(
        corrupted_activation,
        index,
        clean_activation,
):
    """
    Patch activations for a specific head across all positions.
    Used for Q, K, V vectors and attention head outputs (z).

    Args:
        corrupted_activation: Current activation tensor [batch, pos, head, ...]
        index: [layer, head] indices
        clean_activation: Clean activation to patch in
    """
    assert len(index) == 2
    layer, head_index = index
    # Replace activation for this head at all positions
    corrupted_activation[:, :, head_index] = clean_activation[:, :, head_index]
    return corrupted_activation


def layer_head_pattern_patch_setter(
        corrupted_activation,
        index,
        clean_activation,
):
    """
    Patch attention patterns for a specific head across all positions.
    Used for attention pattern matrices.

    Args:
        corrupted_activation: Current pattern tensor [batch, head, dest_pos, src_pos]
        index: [layer, head] indices
        clean_activation: Clean pattern to patch in
    """
    assert len(index) == 2
    layer, head_index = index
    # Replace entire attention pattern for this head
    corrupted_activation[:, head_index, :, :] = clean_activation[:, head_index, :, :]
    return corrupted_activation


def layer_head_pos_pattern_patch_setter(
        corrupted_activation,
        index,
        clean_activation,
):
    """
    Patch attention patterns for a specific head at a specific destination position.

    Args:
        corrupted_activation: Current pattern tensor [batch, head, dest_pos, src_pos]
        index: [layer, head, dest_pos] indices
        clean_activation: Clean pattern to patch in
    """
    assert len(index) == 3
    layer, head_index, dest_pos = index
    # Replace attention pattern for this head at this destination position
    corrupted_activation[:, head_index, dest_pos, :] = clean_activation[:, head_index, dest_pos, :]
    return corrupted_activation


def layer_head_dest_src_pos_pattern_patch_setter(
        corrupted_activation,
        index,
        clean_activation,
):
    """
    Patch a specific entry in the attention pattern matrix.
    Most granular patching - single attention weight.

    Args:
        corrupted_activation: Current pattern tensor [batch, head, dest_pos, src_pos]
        index: [layer, head, dest_pos, src_pos] indices
        clean_activation: Clean pattern to patch in
    """
    assert len(index) == 4
    layer, head_index, dest_pos, src_pos = index
    # Replace single attention weight
    corrupted_activation[:, head_index, dest_pos, src_pos] = clean_activation[
                                                             :, head_index, dest_pos, src_pos
                                                             ]
    return corrupted_activation


# ============================================================================
# PRE-CONFIGURED PATCHING FUNCTIONS
# These are ready-to-use functions for common patching experiments
# ============================================================================

# Residual stream patching functions
get_act_patch_resid_pre = partial(
    generic_activation_patch,
    patch_setter=layer_pos_patch_setter,
    activation_name="resid_pre",  # Residual stream before attention
    index_axis_names=("layer", "pos"),
)

get_act_patch_resid_mid = partial(
    generic_activation_patch,
    patch_setter=layer_pos_patch_setter,
    activation_name="resid_mid",  # Residual stream after attention, before MLP
    index_axis_names=("layer", "pos"),
)

get_act_patch_attn_out = partial(
    generic_activation_patch,
    patch_setter=layer_pos_patch_setter,
    activation_name="attn_out",  # Attention output before residual connection
    index_axis_names=("layer", "pos"),
)

get_act_patch_mlp_out = partial(
    generic_activation_patch,
    patch_setter=layer_pos_patch_setter,
    activation_name="mlp_out",  # MLP output before residual connection
    index_axis_names=("layer", "pos"),
)

# Attention head patching by position
get_act_patch_attn_head_out_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_pos_head_vector_patch_setter,
    activation_name="z",  # Attention head output (after value weighting)
    index_axis_names=("layer", "pos", "head"),
)

get_act_patch_attn_head_q_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_pos_head_vector_patch_setter,
    activation_name="q",  # Query vectors
    index_axis_names=("layer", "pos", "head"),
)

get_act_patch_attn_head_k_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_pos_head_vector_patch_setter,
    activation_name="k",  # Key vectors
    index_axis_names=("layer", "pos", "head"),
)

get_act_patch_attn_head_v_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_pos_head_vector_patch_setter,
    activation_name="v",  # Value vectors
    index_axis_names=("layer", "pos", "head"),
)

# Attention pattern patching
get_act_patch_attn_head_pattern_by_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_pos_pattern_patch_setter,
    activation_name="pattern",  # Attention patterns (where each head looks)
    index_axis_names=("layer", "head_index", "dest_pos"),
)

get_act_patch_attn_head_pattern_dest_src_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_dest_src_pos_pattern_patch_setter,
    activation_name="pattern",  # Individual attention weights
    index_axis_names=("layer", "head_index", "dest_pos", "src_pos"),
)

# Attention head patching across all positions
get_act_patch_attn_head_out_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_vector_patch_setter,
    activation_name="z",  # Attention head output for entire sequence
    index_axis_names=("layer", "head"),
)

get_act_patch_attn_head_q_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_vector_patch_setter,
    activation_name="q",  # Query vectors for entire sequence
    index_axis_names=("layer", "head"),
)

get_act_patch_attn_head_k_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_vector_patch_setter,
    activation_name="k",  # Key vectors for entire sequence
    index_axis_names=("layer", "head"),
)

get_act_patch_attn_head_v_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_vector_patch_setter,
    activation_name="v",  # Value vectors for entire sequence
    index_axis_names=("layer", "head"),
)

get_act_patch_attn_head_pattern_all_pos = partial(
    generic_activation_patch,
    patch_setter=layer_head_pattern_patch_setter,
    activation_name="pattern",  # Entire attention pattern matrix
    index_axis_names=("layer", "head_index"),
)


# ============================================================================
# CONVENIENCE FUNCTIONS FOR COMPREHENSIVE PATCHING
# These run multiple patching experiments at once
# ============================================================================

def get_act_patch_attn_head_all_pos_every(
        model, corrupted_tokens, clean_cache, metric
) -> Float[torch.Tensor, "patch_type layer head"]:
    """
    Run activation patching on all attention head components across all positions.

    Returns results for patching:
    0: Head output (z)
    1: Query vectors (q)
    2: Key vectors (k)
    3: Value vectors (v)
    4: Attention patterns

    Args:
        model: Transformer model
        corrupted_tokens: Corrupted input tokens
        clean_cache: Clean activation cache
        metric: Patching metric function

    Returns:
        Tensor with shape [5, n_layers, n_heads] containing patching results
    """
    act_patch_results: list[torch.Tensor] = []
    act_patch_results.append(
        get_act_patch_attn_head_out_all_pos(model, corrupted_tokens, clean_cache, metric)
    )
    act_patch_results.append(
        get_act_patch_attn_head_q_all_pos(model, corrupted_tokens, clean_cache, metric)
    )
    act_patch_results.append(
        get_act_patch_attn_head_k_all_pos(model, corrupted_tokens, clean_cache, metric)
    )
    act_patch_results.append(
        get_act_patch_attn_head_v_all_pos(model, corrupted_tokens, clean_cache, metric)
    )
    act_patch_results.append(
        get_act_patch_attn_head_pattern_all_pos(model, corrupted_tokens, clean_cache, metric)
    )
    return torch.stack(act_patch_results, dim=0)


def get_act_patch_attn_head_by_pos_every(
        model, corrupted_tokens, clean_cache, metric
) -> Float[torch.Tensor, "patch_type layer pos head"]:
    """
    Run activation patching on all attention head components by position.

    Returns results for patching:
    0: Head output (z)
    1: Query vectors (q)
    2: Key vectors (k)
    3: Value vectors (v)
    4: Attention patterns (reshaped to match other dimensions)

    Args:
        model: Transformer model
        corrupted_tokens: Corrupted input tokens
        clean_cache: Clean activation cache
        metric: Patching metric function

    Returns:
        Tensor with shape [5, n_layers, seq_len, n_heads] containing patching results
    """
    act_patch_results = []
    act_patch_results.append(
        get_act_patch_attn_head_out_by_pos(model, corrupted_tokens, clean_cache, metric)
    )
    act_patch_results.append(
        get_act_patch_attn_head_q_by_pos(model, corrupted_tokens, clean_cache, metric)
    )
    act_patch_results.append(
        get_act_patch_attn_head_k_by_pos(model, corrupted_tokens, clean_cache, metric)
    )
    act_patch_results.append(
        get_act_patch_attn_head_v_by_pos(model, corrupted_tokens, clean_cache, metric)
    )
    # Attention patterns have different shape, so rearrange to match others
    pattern_results = get_act_patch_attn_head_pattern_by_pos(
        model, corrupted_tokens, clean_cache, metric
    )
    act_patch_results.append(einops.rearrange(pattern_results, "batch head pos -> batch pos head"))
    return torch.stack(act_patch_results, dim=0)


def get_act_patch_block_every(
        model, corrupted_tokens, clean_cache, metric
) -> Float[torch.Tensor, "patch_type layer pos"]:
    """
    Run activation patching on all main transformer block components.

    Returns results for patching:
    0: Residual stream pre-attention
    1: Attention output
    2: MLP output

    This helps identify which components of each transformer block are most important
    for the behavior being studied.

    Args:
        model: Transformer model
        corrupted_tokens: Corrupted input tokens
        clean_cache: Clean activation cache
        metric: Patching metric function

    Returns:
        Tensor with shape [3, n_layers, seq_len] containing patching results
    """
    act_patch_results = []
    act_patch_results.append(get_act_patch_resid_pre(model, corrupted_tokens, clean_cache, metric))
    act_patch_results.append(get_act_patch_attn_out(model, corrupted_tokens, clean_cache, metric))
    act_patch_results.append(get_act_patch_mlp_out(model, corrupted_tokens, clean_cache, metric))
    return torch.stack(act_patch_results, dim=0)

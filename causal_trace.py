import argparse
import json
import os
import re
from collections import defaultdict

import numpy
import torch

from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utilities import nethook


class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer. Counts the number
    of layers.

    This class handles the loading and initialization of transformer models,
    automatically detecting the layer structure and storing metadata about
    the model architecture.
    """

    def __init__(
            self,
            model_name=None,
            model=None,
            tokenizer=None,
            low_cpu_mem_usage=False,
            torch_dtype=None,
    ):
        """
        Initialize the model and tokenizer.

        Args:
            model_name: HuggingFace model name (e.g., "gpt2", "EleutherAI/gpt-j-6B")
            model: Pre-loaded model instance (optional)
            tokenizer: Pre-loaded tokenizer instance (optional)
            low_cpu_mem_usage: Whether to use low CPU memory mode during loading
            torch_dtype: Data type for model weights (e.g., torch.float16)
        """
        # Load tokenizer if not provided
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model if not provided
        if model is None:
            assert model_name is not None
            model = AutoModelForCausalLM.from_pretrained(
                model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
            )
            # Freeze all parameters - we're only doing inference, not training
            nethook.set_requires_grad(False, model)
            # Set to evaluation mode and move to GPU
            model.eval().cuda()

        self.tokenizer = tokenizer
        self.model = model

        # Extract layer names by finding modules matching the transformer layer pattern
        # Supports both GPT-2 style (transformer.h.X) and GPT-NeoX style (gpt_neox.layers.X)
        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        """String representation showing model type and layer count."""
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def layername(model, num, kind=None):
    """
    Generate the module name for a specific layer in the model.

    This function constructs the correct module path based on the model architecture
    (GPT-2 vs GPT-NeoX) and the component type (embedding, MLP, attention, or full layer).

    Args:
        model: The transformer model
        num: Layer number (0-indexed)
        kind: Component type - None (full layer), "embed" (embedding), "mlp", or "attn"

    Returns:
        String path to the module (e.g., "transformer.h.5.mlp")
    """
    # Handle GPT-2 style models (transformer.h.X)
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"  # Word token embeddings
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'

    # Handle GPT-NeoX style models (gpt_neox.layers.X)
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"  # Input embeddings
        if kind == "attn":
            kind = "attention"  # GPT-NeoX uses "attention" instead of "attn"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'

    assert False, "unknown transformer structure"


def guess_subject(prompt):
    """
    Attempt to automatically identify the subject of a prompt.

    Uses a regex to find sequences of capitalized words that are likely
    to be proper nouns or entity names, excluding common question words.

    Args:
        prompt: Input text string

    Returns:
        The identified subject string

    Example:
        "The Eiffel Tower is in" -> "Eiffel Tower"
    """
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()


def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
    """
    Visualize causal tracing results as a heatmap.

    Creates a 2D heatmap where:
    - X-axis represents layer numbers
    - Y-axis represents token positions in the input
    - Color intensity shows the impact of restoring that state

    Args:
        result: Dictionary containing tracing results with keys:
                - scores: 2D array of impact scores
                - low_score: Baseline score (fully corrupted)
                - answer: The target answer token
                - kind: Layer type being traced (None, "mlp", or "attn")
                - subject_range: Indices of subject tokens (marked with *)
                - input_tokens: List of input tokens
                - window: Window size (for windowed tracing)
        savepdf: Path to save the plot as PDF (optional)
        title: Custom title for the plot (optional)
        xlabel: Custom x-axis label (optional)
        modelname: Name of the model being analyzed
    """
    differences = result["scores"]
    low_score = result["low_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])

    # Mark subject tokens with asterisks to highlight them
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"

    # Create the plot with Times New Roman font
    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)

        # Create heatmap with color scheme based on layer type:
        # - Purple for full layers (None)
        # - Green for MLPs
        # - Red for attention
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                kind
            ],
            vmin=low_score,  # Set minimum to fully corrupted score
        )

        # Configure axes
        ax.invert_yaxis()  # Put first token at top
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)

        # Set default labels based on context
        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")

        # Add colorbar showing probability scale
        cb = plt.colorbar(h)

        # Apply custom labels if provided
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # Show the target answer token on colorbar
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)

        # Save or display
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def make_inputs(tokenizer, prompts, device="cuda"):
    """
    Convert text prompts into padded token tensors suitable for batch processing.

    Tokenizes prompts and pads them to the same length for efficient batch processing.
    Creates attention masks to indicate which positions are real tokens vs padding.

    Args:
        tokenizer: HuggingFace tokenizer instance
        prompts: List of text strings to tokenize
        device: Target device for tensors (default: "cuda")

    Returns:
        Dictionary with:
        - input_ids: Padded token ID tensor [batch_size, max_length]
        - attention_mask: Binary mask [batch_size, max_length] (1=real token, 0=padding)
    """
    # Tokenize all prompts
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)

    # Determine padding token ID
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0  # Default to 0 if no PAD token exists

    # Left-pad sequences to max length
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]

    # Create attention mask (0 for padding, 1 for real tokens)
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]

    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def decode_tokens(tokenizer, token_array):
    """
    Decode token IDs back to text strings.

    Handles both 1D arrays (single sequence) and 2D arrays (batch of sequences).
    Each token is decoded individually to preserve token boundaries.

    Args:
        tokenizer: HuggingFace tokenizer instance
        token_array: 1D or 2D array of token IDs

    Returns:
        List of decoded token strings, or list of lists for 2D input
    """
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        # Recursively handle batch dimension
        return [decode_tokens(tokenizer, row) for row in token_array]
    # Decode each token separately to maintain boundaries
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
    """
    Find the token indices corresponding to a substring in the decoded text.

    This is useful for identifying which tokens correspond to a particular
    word or phrase (e.g., finding the subject of a sentence).

    Args:
        tokenizer: HuggingFace tokenizer instance
        token_array: 1D array of token IDs
        substring: Text string to locate

    Returns:
        Tuple (start_idx, end_idx) of token positions covering the substring

    Example:
        tokens = [The, Eif, fel, Tower, is]
        substring = "Eiffel Tower"
        returns (1, 4) covering [Eif, fel, Tower]
    """
    # Decode all tokens individually
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)

    # Find character position of substring
    char_loc = whole_string.index(substring)

    # Walk through tokens to find which ones overlap with substring
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


def predict_token(mt, prompts, return_p=False, return_logits_for=None):
    """
    Predict the next token(s) for given prompt(s).

    Runs the model on input prompts and returns predictions. Can optionally
    return probabilities or logits for specific tokens of interest.

    Args:
        mt: ModelAndTokenizer instance
        prompts: List of prompt strings
        return_p: If True, return probabilities of the predicted tokens
        return_logits_for: Token(s) to extract probabilities for. Can be:
                          - A string (e.g., "Paris")
                          - A list of strings (e.g., ["Paris", "London"])
                          - A token ID (int)
                          - A list of token IDs

    Returns:
        result: List of decoded predictions (always returned)
        p: Probabilities of predictions (if return_p=True)
        logits: Probabilities for specified tokens (if return_logits_for provided)

    Example:
        # Simple prediction
        predictions = predict_token(mt, ["The capital of France is"])

        # Get probability of specific answer
        predictions, paris_prob = predict_token(mt, ["The capital of France is"],
                                                return_logits_for="Paris")
    """
    # Tokenize and run model
    inp = make_inputs(mt.tokenizer, prompts)
    preds, p, out = predict_from_input(mt.model, inp, return_logits=True)
    result = [mt.tokenizer.decode(c) for c in preds]

    returns = [result]

    if return_p:
        returns.append(p)

    if return_logits_for is not None:
        # Convert token strings to IDs if needed
        if isinstance(return_logits_for, str):
            # Single token string - encode it
            token_ids = mt.tokenizer.encode(return_logits_for, add_special_tokens=False)
            if len(token_ids) == 0:
                raise ValueError(f"Token '{return_logits_for}' could not be encoded")
            if len(token_ids) > 1:
                print(
                    f"Note: '{return_logits_for}' encodes to {len(token_ids)} tokens: {token_ids}. Returning probabilities for all.")
            token_id = token_ids  # List of token IDs

        elif isinstance(return_logits_for, (list, tuple)):
            # Check if it's a list of strings or IDs
            if all(isinstance(x, str) for x in return_logits_for):
                # List of token strings - encode each
                token_ids = []
                for token_str in return_logits_for:
                    tids = mt.tokenizer.encode(token_str, add_special_tokens=False)
                    if len(tids) == 0:
                        raise ValueError(f"Token '{token_str}' could not be encoded")
                    if len(tids) > 1:
                        print(f"Note: '{token_str}' encodes to {len(tids)} tokens: {tids}. Including all.")
                    token_ids.extend(tids)
                token_id = token_ids
            else:
                # Already a list of token IDs
                token_id = return_logits_for
        else:
            # Single token ID
            token_id = return_logits_for

        # Convert logits to probabilities using softmax over vocabulary
        probs = torch.softmax(out[:, -1], dim=1)

        # Extract probabilities for the specified token(s)
        if isinstance(token_id, (list, tuple)):
            token_probs = probs[:, token_id]
        else:
            token_probs = probs[:, token_id]

        returns.append(token_probs)

    return tuple(returns) if len(returns) > 1 else returns[0]


def predict_from_input(model, inp, return_logits=False):
    """
    Run model inference on tokenized inputs.

    Lower-level function that operates on pre-tokenized input tensors.
    Used by predict_token() and other functions.

    Args:
        model: The language model
        inp: Input dictionary with 'input_ids' and 'attention_mask'
        return_logits: If True, return full logits tensor

    Returns:
        preds: Predicted token IDs (greedy decoding) [batch_size]
        p: Probabilities of predictions [batch_size]
        out: Full logits tensor [batch_size, seq_len, vocab_size] (if return_logits=True)
    """
    # Run forward pass
    out = model(**inp)["logits"]

    # Convert logits to probabilities for last position (next token)
    probs = torch.softmax(out[:, -1], dim=1)

    # Get most likely token (greedy decoding)
    p, preds = torch.max(probs, dim=1)

    if return_logits:
        return preds, p, out
    return preds, p


def collect_embedding_std(mt, subjects):
    """
    Calculate the standard deviation of embeddings across subjects.

    This is used to calibrate the noise level for corruption in causal tracing.
    By measuring natural variation in embeddings, we can add noise at a
    meaningful scale relative to the model's representations.

    Args:
        mt: ModelAndTokenizer instance
        subjects: List of subject strings to analyze

    Returns:
        noise_level: Standard deviation of embedding values (scalar)

    Example:
        subjects = ["Paris", "London", "Tokyo", "Berlin"]
        noise_level = collect_embedding_std(mt, subjects)
        # Use this noise_level in causal tracing experiments
    """
    alldata = []

    # Collect embeddings for all subjects
    for s in subjects:
        inp = make_inputs(mt.tokenizer, [s])
        # Trace the embedding layer to capture input embeddings
        with nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
            mt.model(**inp)
            alldata.append(t.output[0])

    # Concatenate all embeddings and compute standard deviation
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()

    return noise_level

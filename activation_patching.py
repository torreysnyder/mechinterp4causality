import os, re, json
import torch
import numpy as np
from collections import defaultdict
from utilities import nethook
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Dict

from causal_trace import (
    ModelAndTokenizer,
    layername,
    guess_subject,
    plot_trace_heatmap,
)
from causal_trace import (
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)

torch.set_grad_enabled(False)

model_name = "gpt2"  # or "EleutherAI/gpt-j-6B" or "EleutherAI/gpt-neox-20b"
mt = ModelAndTokenizer(
    model_name,
    torch_dtype=torch.float16,
)


def load_template_pairs(path):
    """
    Reads a template file where each *pair* of lines forms:
      line 1 -> clean prompt
      line 2 -> corrupted prompt
    Returns: list of (clean, corrupted) tuples.
    Ignores blank lines.
    """
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) % 2 != 0:
        raise ValueError(f"Expected an even number of lines in {path}, got {len(lines)}.")
    pairs = []
    for i in range(0, len(lines), 2):
        pairs.append((lines[i], lines[i + 1]))
    return pairs


def find_word_token_substitutions(tokenizer, clean_prompt: str, corrupted_prompt: str):
    """
    Find all token-level substitutions needed to transform clean_prompt into corrupted_prompt.

    This function handles multi-token words by finding contiguous spans of differing tokens
    and returns a list of (position, new_token_id) tuples for all positions that differ.

    Args:
        tokenizer: The tokenizer to use
        clean_prompt: Original prompt
        corrupted_prompt: Modified prompt with word substitution(s)

    Returns:
        List of (position, new_token_id) tuples representing all token substitutions needed

    Raises:
        ValueError: If the prompts have different token lengths or no differences found
    """
    clean_ids = tokenizer.encode(clean_prompt, add_special_tokens=False)
    corrupted_ids = tokenizer.encode(corrupted_prompt, add_special_tokens=False)

    if len(clean_ids) != len(corrupted_ids):
        raise ValueError(
            f"Token length mismatch: clean has {len(clean_ids)} tokens, "
            f"corrupted has {len(corrupted_ids)} tokens. "
            f"Word-level substitution must preserve token count."
        )

    # Find all positions where tokens differ
    substitutions = []
    for pos, (clean_tok, corrupt_tok) in enumerate(zip(clean_ids, corrupted_ids)):
        if clean_tok != corrupt_tok:
            substitutions.append((pos, corrupt_tok))

    if not substitutions:
        raise ValueError("No token differences found between clean and corrupted prompts")

    return substitutions


def trace_with_patch(
        model,  # The model
        inp,  # A set of inputs
        states_to_patch,  # A list of (token index, layername) pairs to restore
        answers_t,  # Answer token index to collect
        tokens_to_mix,  # Range of tokens to corrupt (begin, end)
        noise=0.1,  # Level of noise to add
        trace_layers=None,  # List of traced outputs to return
        token_substitutions=None,  # List of (position, new_token_id) tuples
        tokenizer=None,  # Tokenizer for automatic position finding
        **forward_kwargs,  # Extra kwargs passed to model(**inp, **forward_kwargs)
):
    """
    Activation patching with support for token substitution.

    Returns:
        probs: Softmax probability for the answer token
        all_traced: (optional) Stacked activations if trace_layers is not None
    """
    prng = np.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")

    # Handle token substitution if requested
    if token_substitutions is not None:
        # Create corrupted versions by substituting tokens
        clean_input_ids = inp['input_ids'][0:1]  # Keep first element as clean

        # For token substitution, create just 1 corrupted run
        num_corrupted = 1

        if tokenizer is not None:
            print(f"Applying {len(token_substitutions)} token substitution(s):")
            for pos, new_token_id in token_substitutions:
                old_token_id = clean_input_ids[0, pos].item()
                old_token_str = tokenizer.decode([old_token_id])
                new_token_str = tokenizer.decode([new_token_id])
                print(f"  Position {pos}: '{old_token_str}' -> '{new_token_str}'")

        corrupted_inputs = []
        for _ in range(num_corrupted):
            corrupted = clean_input_ids.clone()
            for pos, token_id in token_substitutions:
                corrupted[0, pos] = token_id
            corrupted_inputs.append(corrupted)

        # Rebuild the input batch with clean run first, then corrupted runs
        inp['input_ids'] = torch.cat([clean_input_ids] + corrupted_inputs, dim=0)

        # Expand attention_mask if needed
        if 'attention_mask' in inp:
            clean_mask = inp['attention_mask'][0:1]
            inp['attention_mask'] = clean_mask.repeat(num_corrupted + 1, 1)

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            # This is skipped if token_substitutions is used (unless both are desired)
            if tokens_to_mix is not None and token_substitutions is None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + additional_layers,
            edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp, **forward_kwargs)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


def calculate_hidden_flow(
        mt, prompt, subject, samples=10, noise=0.1, window=10, kind=None,
        token_substitutions=None, target_token=None
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.

    Args:
        mt: ModelAndTokenizer instance
        prompt: Input prompt text
        subject: Subject span to corrupt (for noise) or None (for token substitution)
        samples: Number of samples for noise-based corruption (ignored for token substitution)
        noise: Noise level for corruption (ignored for token substitution)
        window: Window size for windowed tracing
        kind: Type of layers to trace ('mlp', 'attn', or None for all)
        token_substitutions: List of (position, new_token_id) tuples for word-level substitution
        target_token: Specific token to measure probability for (if None, uses predicted token)

    Returns:
        dict with keys:
            scores:           [L x T] tensor (importance by layer/token)
            low_score:        probability under corruption/substitution (float)
            high_score:       base probability under clean input (float)
            input_ids:        tensor of token ids for the clean prompt
            input_tokens:     list of decoded token strings
            subject_range:    (start, end) indices to highlight
            answer:           decoded predicted answer token
            window:           window size used for tracing
            kind:             '' | 'mlp' | 'attn'
    """
    # Create inputs based on corruption method
    if token_substitutions is not None:
        # Token substitution: single, clean input (the actual substitution
        # is applied inside trace_with_patch)
        inp = make_inputs(mt.tokenizer, [prompt])
        tokens_to_mix = None
        e_range = None
    else:
        # Noise-based: multiple samples (first is clean, rest are corrupted)
        inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
        e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
        tokens_to_mix = e_range

    # Base (clean) prediction
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]

    # If target_token is specified, use it instead of the predicted token
    if target_token is not None:
        answer_t = target_token
        # Recalculate base_score for the target token
        with torch.no_grad():
            outputs = mt.model(**inp)
            base_score = torch.softmax(outputs.logits[0, -1, :], dim=0)[answer_t].item()

    [answer] = decode_tokens(mt.tokenizer, [answer_t])

    # Probability under corruption/substitution with no restoration
    low_score = trace_with_patch(
        mt.model,
        inp,
        [],  # no states restored
        answer_t,
        tokens_to_mix,
        noise=noise,
        token_substitutions=token_substitutions,
        tokenizer=mt.tokenizer,
    ).item()

    # Full importance sweep
    if not kind:
        differences = trace_important_states(
            mt.model,
            mt.num_layers,
            inp,
            tokens_to_mix,
            answer_t,
            noise=noise,
            token_substitutions=token_substitutions,
            tokenizer=mt.tokenizer,
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            tokens_to_mix,
            answer_t,
            noise=noise,
            window=window,
            kind=kind,
            token_substitutions=token_substitutions,
            tokenizer=mt.tokenizer,
        )
    differences = differences.detach().cpu()

    # Highlight range for the plot: use the span of substituted tokens
    if token_substitutions is None:
        subject_range = e_range
    else:
        if token_substitutions:
            # Find the contiguous span of substituted positions
            positions = sorted([pos for pos, _ in token_substitutions])
            subject_range = (positions[0], positions[-1] + 1)
        else:
            subject_range = (0, 0)

    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=subject_range,
        answer=answer,
        window=window,
        kind=kind or "",
    )


def _nanpad_to_width(a: np.ndarray, width: int):
    """Pad a 2D array [L, T] with NaNs on the right to width T=width."""
    L, T = a.shape
    if T == width:
        return a
    out = np.full((L, width), np.nan, dtype=float)
    out[:, :T] = a
    return out


def _save_avg_heatmap(avg_matrix: np.ndarray, title: str, outfile: Path):
    """Save an average flow heatmap to file."""
    plt.figure(figsize=(10, 5))
    plt.imshow(avg_matrix, aspect='auto', origin='lower')
    plt.colorbar(label='Avg restoration Δ (prob)')
    plt.xlabel('Token position (index)')
    plt.ylabel('Layer')
    plt.title(title)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=200)
    plt.close()


def load_so_templates(path: Union[str, Path]) -> List[Tuple[str, str]]:
    """
    Reads a templates file where:
      line 1 = clean prompt
      line 2 = corrupted prompt
    and repeats per pair.
    Returns list of (clean, corrupted).
    """
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines()]
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            clean = lines[i]
            bad = lines[i + 1]
            if clean and bad:
                pairs.append((clean, bad))
    return pairs


# ---- Main aggregator ---------------------------------------------------------

def plot_average_flows_over_templates(
        mt,
        templates_file: Union[str, Path],
        samples: int = 10,
        noise: float = 0.1,
        window: int = 10,
        save_dir: Union[str, Path] = "plots_avg",
        modelname: str = None,
        target_token: int = None,
) -> Dict[str, np.ndarray]:
    """
    Generates THREE average plots across all templates:
      1) Hidden state (all)
      2) MLP-only
      3) Attention-only

    - Detects word-level differences and handles multi-token substitutions.
    - Skips templates where token lengths don't match.
    - Handles variable sequence lengths by NaN-padding before averaging.

    Saves:
      <save_dir>/avg_hidden.png
      <save_dir>/avg_mlp.png
      <save_dir>/avg_attn.png

    Returns:
      dict with numpy arrays for each average: {'hidden': [L,T], 'mlp': [L,T], 'attn': [L,T]}
    """
    save_dir = Path(save_dir)
    pairs = load_so_templates(templates_file)

    hidden_mats, mlp_mats, attn_mats = [], [], []
    max_T_hidden = max_T_mlp = max_T_attn = 0
    L_seen = None

    for idx, (clean_prompt, corrupted_prompt) in enumerate(pairs):
        try:
            substitutions = find_word_token_substitutions(mt.tokenizer, clean_prompt, corrupted_prompt)
        except ValueError as e:
            print(f"[avg] Skipping pair #{idx:03d}: {e}")
            continue

        # Hidden / all
        try:
            res_hidden = calculate_hidden_flow(
                mt,
                prompt=clean_prompt,
                subject=None,
                samples=samples,
                noise=noise,
                window=window,
                kind=None,
                token_substitutions=substitutions,
                target_token=target_token,
            )
            H = res_hidden["scores"].detach().cpu().numpy()
            if L_seen is None:
                L_seen = H.shape[0]
            max_T_hidden = max(max_T_hidden, H.shape[1])
            hidden_mats.append(H)
        except Exception as e:
            print(f"[avg] Hidden flow failed on pair #{idx:03d}: {e}")

        # MLP
        try:
            res_mlp = calculate_hidden_flow(
                mt,
                prompt=clean_prompt,
                subject=None,
                samples=samples,
                noise=noise,
                window=window,
                kind="mlp",
                token_substitutions=substitutions,
                target_token=target_token,
            )
            M = res_mlp["scores"].detach().cpu().numpy()
            max_T_mlp = max(max_T_mlp, M.shape[1])
            mlp_mats.append(M)
        except Exception as e:
            print(f"[avg] MLP flow failed on pair #{idx:03d}: {e}")

        # Attention
        try:
            res_attn = calculate_hidden_flow(
                mt,
                prompt=clean_prompt,
                subject=None,
                samples=samples,
                noise=noise,
                window=window,
                kind="attn",
                token_substitutions=substitutions,
                target_token=target_token,
            )
            A = res_attn["scores"].detach().cpu().numpy()
            max_T_attn = max(max_T_attn, A.shape[1])
            attn_mats.append(A)
        except Exception as e:
            print(f"[avg] Attn flow failed on pair #{idx:03d}: {e}")

    if not hidden_mats and not mlp_mats and not attn_mats:
        raise RuntimeError("No valid templates to average. Check your templates file or tokenization.")

    # Pad to common widths and average with NaN-ignoring mean
    def _avg_stack(mats, width):
        if not mats:
            return None
        padded = [_nanpad_to_width(m.astype(float), width) for m in mats]
        stack = np.stack(padded, axis=0)
        with np.errstate(invalid="ignore"):
            avg = np.nanmean(stack, axis=0)
        avg = np.nan_to_num(avg, nan=0.0)
        return avg

    avg_hidden = _avg_stack(hidden_mats, max_T_hidden)
    avg_mlp = _avg_stack(mlp_mats, max_T_mlp)
    avg_attn = _avg_stack(attn_mats, max_T_attn)

    # Save plots
    model_tag = " (%s)" % modelname if modelname else ""
    if avg_hidden is not None:
        _save_avg_heatmap(avg_hidden, "Average Hidden Flow%s" % model_tag, save_dir / "avg_hidden.png")
    if avg_mlp is not None:
        _save_avg_heatmap(avg_mlp, "Average MLP Flow%s" % model_tag, save_dir / "avg_mlp.png")
    if avg_attn is not None:
        _save_avg_heatmap(avg_attn, "Average Attention Flow%s" % model_tag, save_dir / "avg_attn.png")

    # Also save per-layer line plots (mean over token positions)
    if avg_hidden is not None:
        _save_per_layer_line(avg_hidden, "Average Hidden Flow – Per Layer%s" % model_tag,
                             save_dir / "avg_hidden_per_layer.png")
    if avg_mlp is not None:
        _save_per_layer_line(avg_mlp, "Average MLP Flow – Per Layer%s" % model_tag, save_dir / "avg_mlp_per_layer.png")
    if avg_attn is not None:
        _save_per_layer_line(avg_attn, "Average Attention Flow – Per Layer%s" % model_tag,
                             save_dir / "avg_attn_per_layer.png")

    return {"hidden": avg_hidden, "mlp": avg_mlp, "attn": avg_attn}


def trace_important_states(model, num_layers, inp, e_range, answer_t, noise=0.1,
                           token_substitutions=None, tokenizer=None):
    """
    Trace important states across all token positions and layers.

    Args:
        model: The model
        num_layers: Number of layers in the model
        inp: Input dictionary
        e_range: Token range to corrupt (for noise) or None (for token substitution)
        answer_t: Answer token indices
        noise: Noise level (ignored for token substitution)
        token_substitutions: List of (position, new_token_id) tuples
        tokenizer: Tokenizer instance
    """
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                token_substitutions=token_substitutions,
                tokenizer=tokenizer,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
        model, num_layers, inp, e_range, answer_t, kind, window=10, noise=0.1,
        token_substitutions=None, tokenizer=None
):
    """
    Trace important states using a sliding window across layers.

    Args:
        model: The model
        num_layers: Number of layers in the model
        inp: Input dictionary
        e_range: Token range to corrupt (for noise) or None (for token substitution)
        answer_t: Answer token indices
        kind: Type of layers ('mlp', 'attn', or None)
        window: Window size for sliding window
        noise: Noise level (ignored for token substitution)
        token_substitutions: List of (position, new_token_id) tuples
        tokenizer: Tokenizer instance
    """
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model, inp, layerlist, answer_t, tokens_to_mix=e_range, noise=noise,
                token_substitutions=token_substitutions, tokenizer=tokenizer
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def plot_trace_heatmap_with_scores(result, savepdf=None, title=None, xlabel=None, modelname=None):
    """
    Modified version of plot_trace_heatmap that adds score annotations to colorbar.
    """
    import matplotlib.pyplot as plt
    from matplotlib import patches

    differences = result["scores"]
    low_score = result["low_score"]
    high_score = result["high_score"]
    answer = result["answer"]
    kind = result.get("kind", None)
    if kind == "":
        kind = None
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    subject_range = result.get("subject_range", None)

    # Add asterisks to labels in subject_range
    if subject_range is not None:
        start, end = subject_range
        for i in range(start, end):
            if 0 <= i < len(labels):
                labels[i] = labels[i] + "*"

    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "mlp": "Greens", "attn": "Reds"}[
                kind
            ],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, len(labels) - 1, 5)])
        ax.set_xticklabels(list(range(0, len(labels) - 1, 5)))
        ax.set_yticklabels(labels)
        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input", pad=25)
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input", pad=25)
            ax.set_xlabel(f"single restored {kindname} layer within {modelname}")
        cb = plt.colorbar(h)
        cb.set_label(f"p({answer})")

        # Add score annotations next to the colorbar at top and bottom positions
        # Use data coordinates to place them at the actual score values
        cb.ax.text(2.5, high_score, f'High score: {high_score:.3f}',
                   transform=cb.ax.get_yaxis_transform(),
                   fontsize=7, va='bottom', ha='left')
        cb.ax.text(2.5, low_score, f'Low score: {low_score:.3f}',
                   transform=cb.ax.get_yaxis_transform(),
                   fontsize=7, va='top', ha='left')

        if savepdf:
            Path(savepdf).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_hidden_flow(
        mt,
        prompt,
        subject=None,
        samples=10,
        noise=0.1,
        window=10,
        kind=None,
        modelname=None,
        savepdf=None,
        token_substitutions=None,
        target_token=None,
):
    """
    Plot hidden flow heatmap for causal tracing.

    Args:
        mt: ModelAndTokenizer instance
        prompt: Input prompt
        subject: Subject to corrupt (for noise-based) or None
        samples: Number of samples for noise-based corruption
        noise: Noise level
        window: Window size
        kind: Layer type ('mlp', 'attn', or None)
        modelname: Model name for plot title
        savepdf: Path to save PDF
        token_substitutions: List of (position, new_token_id) tuples for word-level substitution
        target_token: Specific token to measure probability for
    """
    if subject is None and token_substitutions is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt, prompt, subject, samples=samples, noise=noise, window=window, kind=kind,
        token_substitutions=token_substitutions, target_token=target_token
    )
    plot_trace_heatmap_with_scores(result, savepdf, modelname=modelname)


def trace_attention_heads(
        model, num_layers, num_heads, inp, e_range, answer_t, noise=0.1,
        token_substitutions=None, tokenizer=None
):
    """
    Trace importance of individual attention heads across all positions.

    Args:
        model: The model
        num_layers: Number of layers
        num_heads: Number of attention heads per layer
        inp: Input dictionary
        e_range: Token range to corrupt (for noise) or None (for token substitution)
        answer_t: Answer token indices
        noise: Noise level
        token_substitutions: List of (position, new_token_id) tuples
        tokenizer: Tokenizer instance

    Returns:
        Tensor of shape [num_positions, num_layers, num_heads] with restoration scores
    """
    ntoks = inp["input_ids"].shape[1]
    table = []

    for tnum in range(ntoks):
        layer_results = []
        for layer in range(num_layers):
            head_results = []
            for head in range(num_heads):
                # Isolate a single attention head using HuggingFace's head_mask.
                mask = torch.zeros((num_layers, num_heads), dtype=torch.float32, device=inp['input_ids'].device)
                mask[layer, head] = 1.0

                r = trace_with_patch(
                    model,
                    inp,
                    [(tnum, layername(model, layer, 'attn'))],
                    answer_t,
                    tokens_to_mix=e_range,
                    noise=noise,
                    token_substitutions=token_substitutions,
                    tokenizer=tokenizer,
                    head_mask=mask,
                )
                head_results.append(r)
            layer_results.append(torch.stack(head_results))
        table.append(torch.stack(layer_results))

    return torch.stack(table)


def calculate_attention_head_flow(
        mt, prompt, subject, samples=10, noise=0.1,
        token_substitutions=None, target_token=None
):
    """
    Calculate attention head activation patching across all positions.

    Returns:
        dict with:
            scores: [num_positions, num_layers, num_heads] tensor
            low_score: probability under corruption
            high_score: probability under clean input
            ... (other metadata)
    """
    # Determine number of attention heads (GPT-2 specific)
    if "gpt2" in mt.model.config._name_or_path.lower():
        num_heads = mt.model.config.n_head
    else:
        num_heads = 12  # default assumption

    # Create inputs
    if token_substitutions is not None:
        inp = make_inputs(mt.tokenizer, [prompt])
        tokens_to_mix = None
        e_range = None
    else:
        inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
        e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
        tokens_to_mix = e_range

    # Base prediction
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]

    if target_token is not None:
        answer_t = target_token
        with torch.no_grad():
            outputs = mt.model(**inp)
            base_score = torch.softmax(outputs.logits[0, -1, :], dim=0)[answer_t].item()

    [answer] = decode_tokens(mt.tokenizer, [answer_t])

    # Low score
    low_score = trace_with_patch(
        mt.model,
        inp,
        [],
        answer_t,
        tokens_to_mix,
        noise=noise,
        token_substitutions=token_substitutions,
        tokenizer=mt.tokenizer,
    ).item()

    # Trace attention heads
    differences = trace_attention_heads(
        mt.model,
        mt.num_layers,
        num_heads,
        inp,
        tokens_to_mix,
        answer_t,
        noise=noise,
        token_substitutions=token_substitutions,
        tokenizer=mt.tokenizer,
    )
    differences = differences.detach().cpu()

    # Subject range
    if token_substitutions is None:
        subject_range = e_range
    else:
        if token_substitutions:
            positions = sorted([pos for pos, _ in token_substitutions])
            subject_range = (positions[0], positions[-1] + 1)
        else:
            subject_range = (0, 0)

    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=subject_range,
        answer=answer,
    )


def plot_attention_head_heatmap_average(results_list, savepdf=None,
                                        title="Attention Head Activation Patching (All Pos)"):
    """
    Plot average attention head activation patching results across all positions.

    Args:
        results_list: List of result dicts from calculate_attention_head_flow
        savepdf: Path to save PDF
        title: Plot title
    """
    import matplotlib.pyplot as plt

    # Average the scores across all templates
    all_scores = []
    for result in results_list:
        scores = result["scores"]  # [num_positions, num_layers, num_heads]
        # Average over positions to get [num_layers, num_heads]
        avg_over_pos = scores.mean(dim=0)
        all_scores.append(avg_over_pos)

    # Stack and average across templates
    stacked = torch.stack(all_scores)
    avg_scores = stacked.mean(dim=0).numpy()  # [num_layers, num_heads]

    # Create plot
    fig, ax = plt.subplots(figsize=(4, 3), dpi=200)

    # Use diverging colormap centered at 0
    vmax = max(abs(avg_scores.min()), abs(avg_scores.max()))
    h = ax.imshow(
        avg_scores,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
        origin="lower"
    )

    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    # Set ticks
    num_layers, num_heads = avg_scores.shape
    ax.set_xticks(range(0, num_heads, 2))
    ax.set_yticks(range(0, num_layers, 2))

    cb = plt.colorbar(h, ax=ax)
    cb.set_label("Avg Δ probability")

    plt.tight_layout()

    if savepdf:
        Path(savepdf).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_all_flow(
    mt,
    prompt,
    subject=None,
    samples=10,
    noise=0.1,
    window=10,
    modelname=None,
    save_prefix=None,
    token_substitutions=None,
    target_token=None,
):
    """
    Plot hidden flow for all layer types (all, mlp, attn).

    If save_prefix is provided, files are saved as:
      {save_prefix}_all.pdf, {save_prefix}_mlp.pdf, {save_prefix}_attn.pdf
    """
    for kind, suffix in [(None, "all"), ("mlp", "mlp"), ("attn", "attn")]:
        savepdf = f"{save_prefix}_{suffix}.pdf" if save_prefix else None
        plot_hidden_flow(
            mt,
            prompt,
            subject=subject,
            samples=samples,
            noise=noise,
            window=window,
            kind=kind,
            modelname=modelname,
            savepdf=savepdf,
            token_substitutions=token_substitutions,
            target_token=target_token,
        )


def _save_per_layer_line(avg_matrix: np.ndarray, title: str, outfile: Path):
    """
    Plot mean importance per layer (averaged over token positions) and save.
    """
    if avg_matrix is None:
        return
    # mean over token dimension (axis=1 → per-layer)
    y = np.nanmean(avg_matrix, axis=1)
    x = np.arange(len(y))

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker='o')
    plt.xlabel('Layer')
    plt.ylabel('Mean restoration Δ (prob)')
    plt.title(title)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=200)
    plt.close()


if __name__ == "__main__":
    # Configure model
    model_name = "gpt2"
    mt = ModelAndTokenizer(model_name, torch_dtype=torch.float16)

    # Load clean/corrupted prompt pairs from lee_templates.txt
    template_path = "lee_templates.txt"
    pairs = load_template_pairs(template_path)

    # Output directory for PDFs
    outdir = Path("plots_from_lee_templates")
    outdir.mkdir(parents=True, exist_ok=True)

    # Store attention head results for averaging
    attention_head_results = []

    # Process each pair
    for idx, (clean_full, corrupted_full) in enumerate(tqdm(pairs, desc="Templates")):
        # Extract the target word (last word) and create prompts without it
        clean_words = clean_full.split()
        corrupted_words = corrupted_full.split()

        if not clean_words or not corrupted_words:
            print(f"[warn] Skipping #{idx:03d}: empty prompt")
            continue

        # Last word is the target - remove it from prompts
        target_word = clean_words[-1]
        clean_prompt = " ".join(clean_words[:-1])
        corrupted_prompt = " ".join(corrupted_words[:-1])

        # Get the token ID for the target word
        # Handle both single-token and multi-token words
        target_tokens = mt.tokenizer.encode(" " + target_word, add_special_tokens=False)
        if len(target_tokens) == 0:
            # Try without leading space
            target_tokens = mt.tokenizer.encode(target_word, add_special_tokens=False)

        if len(target_tokens) == 0:
            print(f"[warn] Skipping #{idx:03d}: could not tokenize target word '{target_word}'")
            continue

        # Use the first token of the target word for probability measurement
        target_token_id = target_tokens[0]

        try:
            # Find word-level token substitutions (handles multi-token words)
            substitutions = find_word_token_substitutions(mt.tokenizer, clean_prompt, corrupted_prompt)
        except ValueError as e:
            print(f"[warn] Skipping #{idx:03d}: {e}")
            continue

        # Build a readable prefix for saved plots
        short_clean = re.sub(r"\W+", "_", clean_prompt.lower()).strip("_")
        short_clean = (short_clean[:60] + "…") if len(short_clean) > 60 else short_clean
        save_prefix = str(outdir / f"{idx:03d}_{short_clean}")

        # Run plots using word-level substitutions and target token
        plot_all_flow(
            mt,
            clean_prompt,
            token_substitutions=substitutions,
            modelname=model_name,
            save_prefix=save_prefix,
            target_token=target_token_id,
        )

        # Calculate attention head results for this template
        try:
            attn_head_result = calculate_attention_head_flow(
                mt,
                prompt=clean_prompt,
                subject=None,
                samples=10,
                noise=0.1,
                token_substitutions=substitutions,
                target_token=target_token_id,
            )
            attention_head_results.append(attn_head_result)
        except Exception as e:
            print(f"[warn] Attention head flow failed on #{idx:03d}: {e}")

    # Generate average attention head plot
    if attention_head_results:
        avg_attn_plot_path = outdir / "average_attention_heads_all_positions.pdf"
        plot_attention_head_heatmap_average(
            attention_head_results,
            savepdf=str(avg_attn_plot_path),
            title="attn_head_out Activation Patching (All Pos)"
        )
        print(f"Average attention head plot saved: {avg_attn_plot_path}")

    # Optionally: average plots across all templates
    # avg = plot_average_flows_over_templates(
    #     mt,
    #     templates_file="lee_templates.txt",
    #     samples=10,
    #     noise=0.1,
    #     window=10,
    #     save_dir="lee_plots_avg",
    #     modelname="gpt2",
    # )
    # print("Average plots saved in: lee_plots_avg/")

    print(f"Done. Individual PDFs saved in: {outdir.resolve()}")

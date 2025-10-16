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
      line 2 -> corrupted prompt (4th token perturbed)
    Returns: list of (clean, corrupted) tuples.
    Ignores blank lines.
    """
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) % 2 != 0:
        raise ValueError(f"Expected an even number of lines in {path}, got {len(lines)}.")
    pairs = []
    for i in range(0, len(lines), 2):
        pairs.append((lines[i], lines[i+1]))
    return pairs


def trace_with_patch(
        model,  # The model
        inp,  # A set of inputs
        states_to_patch,  # A list of (token index, layername) triples to restore
        answers_t,  # Answer probabilities to collect
        tokens_to_mix,  # Range of tokens to corrupt (begin, end)
        noise=0.1,  # Level of noise to add
        trace_layers=None,  # List of traced outputs to return
        token_substitutions=None,  # List of (token_id_to_find, new_token_id) tuples
        tokenizer=None,  # Tokenizer for automatic position finding
):
    """
    Activation patching with support for token substitution.

    Args:
        model: The model to run
        inp: Input dictionary with 'input_ids' and 'attention_mask'
        states_to_patch: List of (token_index, layername) pairs to restore
        answers_t: Answer token indices to collect probabilities for
        tokens_to_mix: Tuple (begin, end) for token range to add noise to
        noise: Level of Gaussian noise to add
        trace_layers: List of layer names to trace activations from
        token_substitutions: List of (old_token_id, new_token_id) tuples OR
                           list of (position, new_token_id) tuples if positions
                           are already known. The function will automatically
                           search for old_token_id in the sequence if needed.
        tokenizer: Tokenizer instance (optional, for debug output)

    Returns:
        probs: Softmax probabilities for answer tokens
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
        # (unlike noise-based corruption which benefits from multiple samples)
        num_corrupted = 1
    else:
        # For noise-based corruption, use the existing batch size
        # The calling code (e.g., calculate_hidden_flow) creates a batch with samples+1 items
        # We don't modify the input in this case
        pass

    if token_substitutions is not None:
        # Convert token_substitutions to position-based format
        position_substitutions = []
        for old_val, new_token_id in token_substitutions:
            # Check if old_val is already a position (int < 1000 is heuristic for position)
            # or if it's a token_id that needs to be found
            if isinstance(old_val, int) and old_val < 100:
                # Assume it's a position
                position_substitutions.append((old_val, new_token_id))
            else:
                # It's a token_id - find its position
                old_token_id = old_val
                matches = (clean_input_ids[0] == old_token_id).nonzero(as_tuple=True)[0]

                if len(matches) == 0:
                    if tokenizer is not None:
                        token_str = tokenizer.decode([old_token_id])
                        raise ValueError(
                            f"Token '{token_str}' (ID {old_token_id}) not found in input sequence"
                        )
                    else:
                        raise ValueError(f"Token ID {old_token_id} not found in input sequence")

                if len(matches) > 1:
                    if tokenizer is not None:
                        token_str = tokenizer.decode([old_token_id])
                        print(
                            f"Warning: Token '{token_str}' (ID {old_token_id}) appears at multiple positions: {matches.tolist()}")
                        print(f"Using first occurrence at position {matches[0].item()}")
                    else:
                        print(f"Warning: Token ID {old_token_id} appears at multiple positions: {matches.tolist()}")
                        print(f"Using first occurrence at position {matches[0].item()}")

                position = matches[0].item()
                position_substitutions.append((position, new_token_id))

                if tokenizer is not None:
                    old_token_str = tokenizer.decode([old_token_id])
                    new_token_str = tokenizer.decode([new_token_id])
                    print(f"Substituting '{old_token_str}' -> '{new_token_str}' at position {position}")

        corrupted_inputs = []
        for _ in range(num_corrupted):
            corrupted = clean_input_ids.clone()
            for pos, token_id in position_substitutions:
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
        outputs_exp = model(**inp)

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
        token_substitutions=None
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
        token_substitutions:
            - None  → use noise-based corruption over `subject`
            - List of tuples for token substitution. Each tuple can be either:
                (position, new_token_id)       # position is an int index in the sequence
             or (old_token_id, new_token_id)   # old_token_id will be searched in the input ids

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
    [answer] = decode_tokens(mt.tokenizer, [answer_t])

    # Probability under corruption/substitution with no restoration
    low_score = trace_with_patch(
        mt.model,
        inp,
        [],                               # no states restored
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

    # Highlight range for the plot:
    # - For noise: use the subject span e_range
    # - For token substitution: try to infer the position of the changed token
    if token_substitutions is None:
        subject_range = e_range
    else:
        pos_to_highlight = None
        if token_substitutions:
            first = token_substitutions[0]
            # Accept either (position, new_id) or (old_id, new_id)
            if isinstance(first[0], int):
                # Position provided directly
                if 0 <= first[0] < inp["input_ids"].shape[1]:
                    pos_to_highlight = int(first[0])
            else:
                # Old token id provided: find it in the clean input_ids
                old_id = first[0]
                seq = inp["input_ids"][0].tolist()
                try:
                    pos_to_highlight = seq.index(old_id)
                except ValueError:
                    pos_to_highlight = None  # couldn't find it; leave empty highlight
        subject_range = (pos_to_highlight, pos_to_highlight + 1) if pos_to_highlight is not None else (0, 0)

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

def _find_first_diff_pos(tokenizer, clean_prompt: str, corrupted_prompt: str):
    """Return (pos, clean_ids, bad_ids) where pos is the first differing token position (or None)."""
    clean_ids = tokenizer.encode(clean_prompt, add_special_tokens=False)
    bad_ids   = tokenizer.encode(corrupted_prompt, add_special_tokens=False)
    pos = None
    for i, (a, b) in enumerate(zip(clean_ids, bad_ids)):
        if a != b:
            pos = i
            break
    return pos, clean_ids, bad_ids


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
            bad   = lines[i + 1]
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
) -> Dict[str, np.ndarray]:
    """
    Generates THREE average plots across all templates:
      1) Hidden state (all)
      2) MLP-only
      3) Attention-only

    - Detects the correct substitution position via first token diff under BPE.
    - Skips templates where there is no 1↔1 token difference.
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
        pos, clean_ids, bad_ids = _find_first_diff_pos(mt.tokenizer, clean_prompt, corrupted_prompt)

        if pos is None or pos >= len(clean_ids) or pos >= len(bad_ids):
            print("[avg] Skipping pair #%03d: no single-position token diff after BPE." % idx)
            continue

        substitution = [(pos, bad_ids[pos])]

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
                token_substitutions=substitution,
            )
            H = res_hidden["scores"].detach().cpu().numpy()
            if L_seen is None:
                L_seen = H.shape[0]
            max_T_hidden = max(max_T_hidden, H.shape[1])
            hidden_mats.append(H)
        except Exception as e:
            print("[avg] Hidden flow failed on pair #%03d: %s" % (idx, e))

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
                token_substitutions=substitution,
            )
            M = res_mlp["scores"].detach().cpu().numpy()
            max_T_mlp = max(max_T_mlp, M.shape[1])
            mlp_mats.append(M)
        except Exception as e:
            print("[avg] MLP flow failed on pair #%03d: %s" % (idx, e))

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
                token_substitutions=substitution,
            )
            A = res_attn["scores"].detach().cpu().numpy()
            max_T_attn = max(max_T_attn, A.shape[1])
            attn_mats.append(A)
        except Exception as e:
            print("[avg] Attn flow failed on pair #%03d: %s" % (idx, e))

    if not hidden_mats and not mlp_mats and not attn_mats:
        raise RuntimeError("No valid templates to average. Check your templates file or tokenization diffs.")

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
    avg_mlp    = _avg_stack(mlp_mats,    max_T_mlp)
    avg_attn   = _avg_stack(attn_mats,   max_T_attn)

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
        _save_per_layer_line(avg_hidden, "Average Hidden Flow — Per Layer%s" % model_tag, save_dir / "avg_hidden_per_layer.png")
    if avg_mlp is not None:
        _save_per_layer_line(avg_mlp,    "Average MLP Flow — Per Layer%s" % model_tag,    save_dir / "avg_mlp_per_layer.png")
    if avg_attn is not None:
        _save_per_layer_line(avg_attn,   "Average Attention Flow — Per Layer%s" % model_tag, save_dir / "avg_attn_per_layer.png")


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
        token_substitutions: List of (old_token_id, new_token_id) tuples
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
        token_substitutions: List of (old_token_id, new_token_id) tuples
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
        token_substitutions: List of (old_token_id, new_token_id) tuples for token substitution
    """
    if subject is None and token_substitutions is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt, prompt, subject, samples=samples, noise=noise, window=window, kind=kind,
        token_substitutions=token_substitutions
    )
    plot_trace_heatmap(result, savepdf, modelname=modelname)


def plot_all_flow(mt, prompt, subject=None, noise=0.1, modelname=None,
                  token_substitutions=None, save_prefix=None):
    """
    Plot hidden flow for all layer types (all, mlp, attn).

    If save_prefix is provided, files are saved as:
      {save_prefix}_all.pdf, {save_prefix}_mlp.pdf, {save_prefix}_attn.pdf
    """
    for kind, suffix in [(None, "all"), ("mlp", "mlp"), ("attn", "attn")]:
        savepdf = f"{save_prefix}_{suffix}.pdf" if save_prefix else None
        plot_hidden_flow(
            mt, prompt, subject, modelname=modelname, noise=noise, kind=kind,
            token_substitutions=token_substitutions, savepdf=savepdf
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
    model_name = "gpt2"  # keep as-is or change
    mt = ModelAndTokenizer(model_name, torch_dtype=torch.float16)

    # Load clean/corrupted prompt pairs
    template_path = "lee_templates.txt"   # adjust if needed
    pairs = load_template_pairs(template_path)

    # Output directory for PDFs
    outdir = Path("plots_from_lee_templates")
    outdir.mkdir(parents=True, exist_ok=True)

    # Process each pair
    for idx, (clean_prompt, corrupted_prompt) in enumerate(tqdm(pairs, desc="Templates")):
        # Tokenize both prompts (no special tokens)
        clean_ids = mt.tokenizer.encode(clean_prompt, add_special_tokens=False)
        bad_ids   = mt.tokenizer.encode(corrupted_prompt, add_special_tokens=False)

        # Find the first differing token position
        pos = None
        for i, (a, b) in enumerate(zip(clean_ids, bad_ids)):
            if a != b:
                pos = i
                break

        # Safety checks
        if pos is None:
            print(f"[warn] Skipping #{idx:03d}: no token-level difference after BPE tokenization.")
            continue

        # Only handle simple 1↔1 token substitutions for robustness
        if pos >= len(clean_ids) or pos >= len(bad_ids):
            print(f"[warn] Skipping #{idx:03d}: mismatch not 1-to-1 at position {pos}.")
            continue

        # Build a readable prefix for saved plots
        short_clean = re.sub(r"\W+", "_", clean_prompt.lower()).strip("_")
        short_clean = (short_clean[:60] + "…") if len(short_clean) > 60 else short_clean
        save_prefix = str(outdir / f"{idx:03d}_{short_clean}")

        # Run plots using a position-based substitution
        # (trace_with_patch already treats small ints as positions)
        plot_all_flow(
            mt,
            clean_prompt,
            token_substitutions=[(pos, bad_ids[pos])],  # <-- position-based, not token-id search
            modelname=model_name,
            save_prefix=save_prefix,
        )
    avg = plot_average_flows_over_templates(
            mt,
            templates_file="lee_templates.txt",
            samples=10,
            noise=0.1,
            window=10,
            save_dir="lee_plots_avg",
            modelname="gpt2",  # or whatever you pass elsewhere
        )
    print("Average plots saved in: plots_avg/")

    print(f"Done. PDFs saved in: {outdir.resolve()}")

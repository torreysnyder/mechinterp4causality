import os, re, json
import torch, numpy
from collections import defaultdict
from utilities import nethook
from pathlib import Path
from tqdm import tqdm

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
    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
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
        token_substitutions: List of (old_token_id, new_token_id) tuples for token substitution.
                           If provided, uses token substitution instead of noise-based corruption.
    """
    # Create inputs based on corruption method
    if token_substitutions is not None:
        # Token substitution: single input
        inp = make_inputs(mt.tokenizer, [prompt])
        tokens_to_mix = None
    else:
        # Noise-based: multiple samples
        inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
        e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
        tokens_to_mix = e_range

    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])

    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, tokens_to_mix, noise=noise,
        token_substitutions=token_substitutions, tokenizer=mt.tokenizer
    ).item()

    if not kind:
        differences = trace_important_states(
            mt.model, mt.num_layers, inp, tokens_to_mix, answer_t,
            noise=noise, token_substitutions=token_substitutions, tokenizer=mt.tokenizer
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
            tokenizer=mt.tokenizer
        )
    differences = differences.detach().cpu()

    # Give an empty highlight range when using token substitution
    subject_range = e_range if token_substitutions is None else (0, 0)

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

if __name__ == "__main__":
    # Configure model
    model_name = "gpt2"  # keep as-is or change
    mt = ModelAndTokenizer(model_name, torch_dtype=torch.float16)

    # Load clean/corrupted prompt pairs
    template_path = "templates.txt"   # adjust if needed
    pairs = load_template_pairs(template_path)

    # Output directory for PDFs
    outdir = Path("plots_from_templates")
    outdir.mkdir(parents=True, exist_ok=True)

    # Process each pair
    for idx, (clean_prompt, corrupted_prompt) in enumerate(tqdm(pairs, desc="Templates")):
        # Tokenize both prompts (no special tokens)
        clean_ids = mt.tokenizer.encode(clean_prompt, add_special_tokens=False)
        bad_ids   = mt.tokenizer.encode(corrupted_prompt, add_special_tokens=False)

        # Safety checks
        if len(clean_ids) < 4 or len(bad_ids) < 4:
            print(f"[warn] Skipping #{idx:03d}: fewer than 4 tokens after tokenization.")
            continue

        # We assume the 4th token (index 3) is the perturbed one by construction.
        # Create a token_substitutions spec that swaps the 4th token.
        old_token_id = clean_ids[3]
        new_token_id = bad_ids[3]

        # Build a readable prefix for saved plots
        # e.g., 000_if_alice_is_cold...
        short_clean = re.sub(r"\W+", "_", clean_prompt.lower()).strip("_")
        short_clean = (short_clean[:60] + "…") if len(short_clean) > 60 else short_clean
        save_prefix = str(outdir / f"{idx:03d}_{short_clean}")

        # Run plots: clean prompt + token substitution to the corrupted 4th token
        plot_all_flow(
            mt,
            clean_prompt,
            token_substitutions=[(old_token_id, new_token_id)],
            modelname=model_name,
            save_prefix=save_prefix,
        )

    print(f"Done. PDFs saved in: {outdir.resolve()}")

"""
diagnostic_head_patching.py

Runs the activation-patching sign diagnostic on a single prompt pair.

For each token position and layer 0, it compares:
  (A) low_score          — p(answer) with full corruption, no patching at all.
  (B) full_layer_score   — p(answer) when the entire layer-0 attn output is
                           patched from the clean run (no head isolation hook).
                           Computed via trace_with_patch directly.
  (C) per_head_scores    — p(answer) when only one head's output is patched,
                           using the outer nethook isolator from trace_attention_heads.

If the outer isolator hook is corrupting the clean reference activations, we
expect:
  - (B) >> low_score  (full-layer patch restores the prediction strongly)
  - (C) ≈ low_score or < low_score for all heads (per-head patch does nothing
    or hurts, because the clean reference it patches FROM is already zeroed out)

If patching is working correctly, the sum of per-head Δp values across all
heads in a layer should roughly match the full-layer Δp (up to interaction
effects), and the signs should be consistent.

Usage:
    python diagnostic_head_patching.py
    (Hydra config is picked up automatically from causal_transformers/config)
"""

import re
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from transformer_lens.utilities import nethook
import hydra

from causal_transformers.utils import model_utils
from causal_transformers.dataset.preprocessor import Preprocessor

from causal_trace import layername
from causal_trace import make_inputs, decode_tokens, find_token_range

# Import all patching utilities from the main script.
from activation_patching import (
    load_finetuned_model,
    load_template_pairs,
    find_word_token_substitutions,
    trace_with_patch,
    trace_attention_heads,
)

torch.set_grad_enabled(False)

# ── Diagnostic layer / number of heads to probe ───────────────────────────────
DIAG_LAYER = 0       # layer whose heads we want to inspect
N_DIAG_PAIRS = 3     # number of prompt pairs to run the diagnostic on


def run_diagnostic_on_pair(mt, clean_prompt, substitutions, target_tokens, pair_idx):
    """
    For a single (clean_prompt, substitutions, target_tokens) triple, print a
    table comparing low_score, the full layer-0 attn patch score, and the
    per-head patch scores for every position × head in layer 0.
    """
    print(f"\n{'='*70}")
    print(f"Pair #{pair_idx:03d}  prompt: '{clean_prompt[:80]}'")
    print(f"  target token id : {target_tokens}")
    print(f"{'='*70}")

    # Determine num_heads from the underlying HookedTransformer config.
    if hasattr(mt.model, 'cfg') and hasattr(mt.model.cfg, 'n_heads'):
        num_heads = mt.model.cfg.n_heads
    else:
        num_heads = 8  # fallback

    inp = make_inputs(mt.tokenizer, [clean_prompt])
    ntoks = inp["input_ids"].shape[1]

    # ── A: low_score ──────────────────────────────────────────────────────────
    low_score = trace_with_patch(
        mt.model, inp,
        states_to_patch=[],
        answers_t=target_tokens,
        tokens_to_mix=None,
        token_substitutions=substitutions,
        tokenizer=mt.tokenizer,
    ).item()

    print(f"\n[A] low_score (no patching)  = {low_score:.6f}")

    # ── B: full layer-0 attn patch, NO isolator hook ──────────────────────────
    # Patch every token position at layer 0 attn simultaneously, with no outer
    # nethook hook active.  This is the ground-truth full-layer restoration.
    full_layer_states = [
        (tnum, layername(mt.model, DIAG_LAYER, 'attn'))
        for tnum in range(ntoks)
    ]
    full_layer_score = trace_with_patch(
        mt.model, inp,
        states_to_patch=full_layer_states,
        answers_t=target_tokens,
        tokens_to_mix=None,
        token_substitutions=substitutions,
        tokenizer=mt.tokenizer,
    ).item()

    full_layer_delta = full_layer_score - low_score
    print(f"[B] full layer-{DIAG_LAYER} attn patch  = {full_layer_score:.6f}  "
          f"(Δp = {full_layer_delta:+.6f})")

    # ── C: per-head patch scores WITH the outer isolator hook ─────────────────
    # This replicates exactly what trace_attention_heads does.
    head_hook_name = f"blocks.{DIAG_LAYER}.attn.hook_z"
    raw_model = mt.model.model if hasattr(mt.model, 'model') else mt.model

    print(f"\n[C] per-head patch scores for layer {DIAG_LAYER} "
          f"(averaged over {ntoks} positions):")
    print(f"  {'Head':>4}  {'raw_p':>10}  {'Δp':>10}  {'consistent_sign':>16}")

    per_head_deltas = []

    for head in range(num_heads):
        # Patch all positions for this head simultaneously (mirrors the tnum
        # loop in trace_attention_heads, but condensed for the diagnostic).
        head_scores_per_pos = []

        def make_isolator(target_head):
            def isolate(x, layer_name):
                mask = torch.zeros(x.shape[-2], dtype=x.dtype, device=x.device)
                mask[target_head] = 1.0
                x[1:] = x[1:] * mask[None, None, :, None]
                return x
            return isolate

        isolator = make_isolator(head)

        for tnum in range(ntoks):
            with nethook.TraceDict(
                raw_model,
                [head_hook_name],
                edit_output=isolator,
            ):
                r = trace_with_patch(
                    mt.model, inp,
                    states_to_patch=[(tnum, layername(mt.model, DIAG_LAYER, 'attn'))],
                    answers_t=target_tokens,
                    tokens_to_mix=None,
                    token_substitutions=substitutions,
                    tokenizer=mt.tokenizer,
                ).item()
            head_scores_per_pos.append(r)

        avg_raw = float(np.mean(head_scores_per_pos))
        delta = avg_raw - low_score
        per_head_deltas.append(delta)

        # Sign consistency: does this head's sign agree with full-layer Δp?
        consistent = (delta >= 0) == (full_layer_delta >= 0)
        consistent_str = "✓ yes" if consistent else "✗ NO"

        print(f"  {head:>4}  {avg_raw:>10.6f}  {delta:>+10.6f}  {consistent_str:>16}")

    # ── Summary ───────────────────────────────────────────────────────────────
    sum_head_deltas = sum(per_head_deltas)
    print(f"\n  Sum of per-head Δp across heads : {sum_head_deltas:+.6f}")
    print(f"  Full-layer Δp                   : {full_layer_delta:+.6f}")
    ratio = sum_head_deltas / full_layer_delta if abs(full_layer_delta) > 1e-8 else float('nan')
    print(f"  Ratio (sum_heads / full_layer)  : {ratio:.3f}  "
          f"(expected ~1 if no interaction effects; large deviation suggests hook interference)")

    all_negative = all(d < 0 for d in per_head_deltas)
    if full_layer_delta > 0.01 and all_negative:
        print("\n  *** DIAGNOSIS: HOOK INTERFERENCE DETECTED ***")
        print("  Full-layer patch is POSITIVE but all per-head patches are NEGATIVE.")
        print("  The outer isolator hook is likely zeroing the clean reference")
        print("  activations (x[0]) before trace_with_patch can read them.")
    elif all(d < 0 for d in per_head_deltas) and full_layer_delta < 0:
        print("\n  *** DIAGNOSIS: CONSISTENT NEGATIVE — LIKELY GENUINE ***")
        print("  Both full-layer and per-head patches are negative.")
        print("  These heads carry intervention-specific signal; patching in")
        print("  clean activations suppresses that signal, hurting performance.")
    else:
        print("\n  *** DIAGNOSIS: MIXED SIGNS — patching appears to be working correctly ***")


@hydra.main(config_path="causal_transformers/config", config_name="config", version_base=None)
def main(cfg):
    # ── Model loading ─────────────────────────────────────────────────────────
    checkpoint_path = (
        "causal_transformers/checkpoints/gaussian/gaussian_dataset_v6/"
        "hooked_transformer_deep/instance3/checkpoint_0300.pth"
    )
    print(f"Loading model from: {checkpoint_path}")
    model = load_finetuned_model(checkpoint_path, cfg)

    class FineTunedModelWrapper:
        def __init__(self, model, cfg):
            self.model = model
            self.num_layers = cfg.model.hooked_transformer.n_layers
            self.tokenizer = Preprocessor(cfg)

        def __call__(self, input_ids=None, attention_mask=None, **kwargs):
            if input_ids is None:
                raise ValueError("input_ids is required")
            vocab_size = self.model.cfg.d_vocab
            if (input_ids < 0).any() or (input_ids >= vocab_size).any():
                raise ValueError(
                    f"Token IDs out of range [0, {vocab_size}): "
                    f"[{input_ids.min()}, {input_ids.max()}]"
                )
            logits = self.model(input_ids)
            return {"logits": logits}

    mt = FineTunedModelWrapper(model, cfg)
    print(f"Model loaded. Layers: {mt.num_layers}")

    # ── Load prompt pairs ─────────────────────────────────────────────────────
    template_path = "causal_transformers/inference/intervention.txt"
    pairs = load_template_pairs(template_path)

    n_run = 0
    for idx, (clean_full, corrupted_full) in enumerate(pairs):
        if n_run >= N_DIAG_PAIRS:
            break

        clean_words = clean_full.split()
        corrupted_words = corrupted_full.split()

        if len(clean_words) < 2:
            continue

        predicted_token_str = clean_words[-1]
        clean_prompt = " ".join(clean_words[:-1])
        corrupted_prompt = " ".join(corrupted_words[:-1])

        if not hasattr(mt.tokenizer, "vocab") or predicted_token_str not in mt.tokenizer.vocab:
            print(f"[warn] Skipping #{idx:03d}: '{predicted_token_str}' not in vocab")
            continue

        target_tokens = mt.tokenizer.vocab[predicted_token_str]

        try:
            substitutions = find_word_token_substitutions(
                mt.tokenizer, clean_prompt, corrupted_prompt
            )
        except ValueError as e:
            print(f"[warn] Skipping #{idx:03d}: {e}")
            continue

        run_diagnostic_on_pair(mt, clean_prompt, substitutions, target_tokens, idx)
        n_run += 1

    print(f"\n\nDiagnostic complete. Ran on {n_run} pair(s).")


if __name__ == "__main__":
    main()

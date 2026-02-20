import re
import torch
import numpy as np
from collections import defaultdict
from transformer_lens.utilities import nethook
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Dict
import hydra
from omegaconf import OmegaConf

from causal_transformers.utils import model_utils, checkpoint_utils
from causal_transformers.dataset.preprocessor import Preprocessor

from causal_trace import (
    ModelAndTokenizer,
    layername,
    guess_subject,
)
from causal_trace import (
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_from_input,
)

torch.set_grad_enabled(False)


def load_finetuned_model(checkpoint_path, cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_utils.get_hooked_transformer_model(cfg, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from checkpoint: {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['stats']['train']['loss']:.6f}")
    return model


@hydra.main(config_path="causal_transformers/config", config_name="config", version_base=None)
def initialize_model(cfg):
    checkpoint_path = "causal_transformers/checkpoints/gaussian/gaussian_dataset_v6/hooked_transformer_deep/instance3/checkpoint_0300.pth"
    model = load_finetuned_model(checkpoint_path, cfg)

    class FineTunedModelWrapper:
        def __init__(self, model, cfg):
            self.model = model
            self.num_layers = cfg.model.hooked_transformer.n_layers
            self.tokenizer = Preprocessor(cfg)

        def __call__(self, input_ids=None, attention_mask=None, **kwargs):
            """
            Make the wrapper compatible with HuggingFace-style calling convention.
            HookedTransformer expects just input_ids as positional arg, not keyword args.
            """
            if input_ids is None:
                raise ValueError("input_ids is required")

            # Validate token IDs are in valid range
            vocab_size = self.model.cfg.d_vocab
            if (input_ids < 0).any() or (input_ids >= vocab_size).any():
                raise ValueError(
                    f"Token IDs must be in range [0, {vocab_size}), got values in range [{input_ids.min()}, {input_ids.max()}]")

            # HookedTransformer returns logits directly, wrap in dict for compatibility
            logits = self.model(input_ids)
            return {"logits": logits}

    return FineTunedModelWrapper(model, cfg)


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

    Preprocessor tokenization is whitespace-splitting, and each space-separated word is a single
    atomic token in the vocabulary. For that case we:
      - tokenize via tokenizer.tokenize(prompt)
      - map tokens to ids via tokenizer.vocab[token]

    For compatibility with HuggingFace tokenizers, we fall back to encode().
    """
    # --- Preprocessor path ---------------------------------------------------
    is_preproc = (
        hasattr(tokenizer, "tokenize")
        and hasattr(tokenizer, "vocab")
        and isinstance(getattr(tokenizer, "vocab"), dict)
    )
    if is_preproc:
        clean_tokens = tokenizer.tokenize(clean_prompt)
        corrupted_tokens = tokenizer.tokenize(corrupted_prompt)

        if len(clean_tokens) != len(corrupted_tokens):
            raise ValueError(
                f"Token length mismatch: clean has {len(clean_tokens)} tokens, "
                f"corrupted has {len(corrupted_tokens)} tokens. "
                f"Word-level substitution must preserve token count."
            )

        substitutions = []
        for pos, (ctok, btok) in enumerate(zip(clean_tokens, corrupted_tokens)):
            if ctok != btok:
                if btok not in tokenizer.vocab:
                    raise KeyError(f"Corrupted token '{btok}' not in Preprocessor vocab.")
                substitutions.append((pos, tokenizer.vocab[btok]))

        if not substitutions:
            raise ValueError("No token differences found between clean and corrupted prompts")

        return substitutions

    # --- HuggingFace fallback ------------------------------------------------
    clean_ids = tokenizer.encode(clean_prompt, add_special_tokens=False)
    corrupted_ids = tokenizer.encode(corrupted_prompt, add_special_tokens=False)

    if len(clean_ids) != len(corrupted_ids):
        raise ValueError(
            f"Token length mismatch: clean has {len(clean_ids)} tokens, "
            f"corrupted has {len(corrupted_ids)} tokens. "
            f"Word-level substitution must preserve token count."
        )

    substitutions = [
        (pos, c_id)
        for pos, (a_id, c_id) in enumerate(zip(clean_ids, corrupted_ids))
        if a_id != c_id
    ]
    if not substitutions:
        raise ValueError("No token differences found between clean and corrupted prompts")
    return substitutions


def trace_with_patch(
        model,  # The model
        inp,  # A set of inputs
        states_to_patch,  # A list of (token index, layername) pairs to restore
        answers_t,  # Answer token ID or list of token IDs to collect
        tokens_to_mix,  # Range of tokens to corrupt (begin, end)
        noise=0.1,  # Level of noise to add
        trace_layers=None,  # List of traced outputs to return
        token_substitutions=None,  # List of (position, new_token_id) tuples
        tokenizer=None,  # Tokenizer for automatic position finding
        **forward_kwargs,  # Extra kwargs passed to model(**inp, **forward_kwargs) for HF models
):
    """
    Activation patching with support for token substitution and multi-token targets.

    Compatible with:
      - HuggingFace GPT-2 style models (model(**inp) -> outputs.logits)
      - TransformerLens HookedTransformer (model(tokens, return_type="logits") -> logits tensor)

    Args:
        answers_t: Either a single token ID (int) or a list of token IDs for multi-token targets

    Returns:
        probs: Softmax probability for the answer token(s)
               - Single float tensor if answers_t is int
               - Float tensor (joint probability) if answers_t is list
        all_traced: (optional) Stacked activations if trace_layers is not None
    """
    prng = np.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    # -------------------------
    # Model IO adapters
    # -------------------------
    def _is_hooked_transformer(m):
        # TransformerLens HookedTransformer signature
        return hasattr(m, "blocks") and hasattr(m, "embed")

    def _strip_hf_only_keys(inp_dict):
        # HookedTransformer doesn't accept attention_mask, token_type_ids, etc.
        hf_only = {"attention_mask", "token_type_ids", "position_ids"}
        return {k: v for k, v in inp_dict.items() if k not in hf_only}

    def _forward(m, inp_dict, fw_kwargs):
        """
        Returns model outputs in a way compatible with both HF and TransformerLens.
        """
        if _is_hooked_transformer(m):
            tokens = inp_dict["input_ids"]  # HF-shaped dict uses input_ids
            # Ignore HF-only kwargs, force logits
            return m(tokens, return_type="logits")
        else:
            return m(**inp_dict, **fw_kwargs)

    def _get_logits(outputs):
        """
        HF: outputs.logits
        TL: outputs is already a logits tensor
        """
        return outputs.logits if hasattr(outputs, "logits") else outputs

    # Determine if we have multi-token target
    is_multitoken = isinstance(answers_t, (list, tuple))
    target_tokens = answers_t if is_multitoken else [answers_t]

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
                old_token_str = decode_tokens(tokenizer, [old_token_id])[0] if tokenizer is not None else str(old_token_id)
                new_token_str = decode_tokens(tokenizer, [new_token_id])[0] if tokenizer is not None else str(new_token_id)
                print(f"  Position {pos}: '{old_token_str}' -> '{new_token_str}'")

        corrupted_inputs = []
        for _ in range(num_corrupted):
            corrupted = clean_input_ids.clone()
            for pos, token_id in token_substitutions:
                corrupted[0, pos] = token_id
            corrupted_inputs.append(corrupted)

        # Rebuild the input batch with clean run first, then corrupted runs
        inp['input_ids'] = torch.cat([clean_input_ids] + corrupted_inputs, dim=0)

        # Expand attention_mask if needed (HF path only; stripped for TL)
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

    # For multi-token targets, we need to autoregressively compute probabilities
    if is_multitoken:
        token_probs = []
        log_prob_sum = 0.0
        current_inp = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inp.items()}

        additional_layers = [] if trace_layers is None else trace_layers

        all_traced = None

        for i, target_token_id in enumerate(target_tokens):
            # Choose appropriate input dict depending on model type
            current_inp_run = _strip_hf_only_keys(current_inp) if _is_hooked_transformer(model) else current_inp

            # With the patching rules defined, run the patched model in inference.
            with torch.no_grad(), nethook.TraceDict(
                    model,
                    [embed_layername] + list(patch_spec.keys()) + additional_layers,
                    edit_output=patch_rep,
            ) as td:
                outputs_exp = _forward(model, current_inp_run, forward_kwargs)

            logits = _get_logits(outputs_exp)

            # We report softmax probabilities for the corrupted runs [1:]
            probs = torch.softmax(logits[1:, -1, :], dim=1).mean(dim=0)[target_token_id]
            token_probs.append(probs.item())
            log_prob_sum += torch.log(probs + 1e-10).item()

            # If tracing all layers on first token, collect activations
            if trace_layers is not None and i == 0:
                all_traced = torch.stack(
                    [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
                )

            # Append this token to the sequence for next iteration (if not last token)
            if i < len(target_tokens) - 1:
                # Extend all sequences in the batch
                new_token = torch.tensor([[target_token_id]], device=current_inp['input_ids'].device)
                new_token = new_token.repeat(current_inp['input_ids'].shape[0], 1)
                current_inp['input_ids'] = torch.cat([current_inp['input_ids'], new_token], dim=1)

                if 'attention_mask' in current_inp:
                    new_mask = torch.ones(
                        (current_inp['attention_mask'].shape[0], 1),
                        device=current_inp['attention_mask'].device
                    )
                    current_inp['attention_mask'] = torch.cat([current_inp['attention_mask'], new_mask], dim=1)

        # Return joint probability as a tensor (not a float)
        joint_prob = torch.tensor(np.exp(log_prob_sum))

        if trace_layers is not None:
            return joint_prob, all_traced
        return joint_prob

    else:
        # Single token case (original behavior)
        inp_run = _strip_hf_only_keys(inp) if _is_hooked_transformer(model) else inp

        with torch.no_grad(), nethook.TraceDict(
                model,
                [embed_layername] + list(patch_spec.keys()) + ([] if trace_layers is None else trace_layers),
                edit_output=patch_rep,
        ) as td:
            outputs_exp = _forward(model, inp_run, forward_kwargs)

        logits = _get_logits(outputs_exp)

        # We report softmax probabilities for the answers_t token predictions of interest.
        probs = torch.softmax(logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

        # If tracing all layers, collect all activations together to return.
        if trace_layers is not None:
            all_traced = torch.stack(
                [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
            )
            return probs, all_traced

        return probs



def calculate_hidden_flow(
        mt, prompt, subject, samples=10, noise=0.1, window=10, kind=None,
        token_substitutions=None, target_tokens=None
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
        target_tokens: Single token ID (int) or list of token IDs for multi-token targets

    Returns:
        dict with keys:
            scores:           [L x T] tensor (importance by layer/token)
            low_score:        probability under corruption/substitution (float)
            high_score:       base probability under clean input (float)
            input_ids:        tensor of token ids for the clean prompt
            input_tokens:     list of decoded token strings
            subject_range:    (start, end) indices to highlight
            answer:           decoded predicted answer token(s)
            window:           window size used for tracing
            kind:             '' | 'mlp' | 'attn'
            is_multitoken:    whether target is multi-token (bool)
            token_probs:      list of individual token probabilities (if multi-token)
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

    # Determine if we have multi-token target
    is_multitoken = isinstance(target_tokens, (list, tuple))

    # Base (clean) prediction
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt, inp)]

    # If target_tokens is specified, use it instead of the predicted token
    if target_tokens is not None:
        if is_multitoken:
            # Multi-token case: compute joint probability
            token_probs_clean = []
            log_prob_sum = 0.0
            current_inp = {k: v[0:1].clone() if torch.is_tensor(v) else v for k, v in inp.items()}

            with torch.no_grad():
                for i, target_token_id in enumerate(target_tokens):
                    outputs = mt(**current_inp)
                    probs = torch.softmax(outputs["logits"][0, -1, :], dim=0)
                    prob = probs[target_token_id].item()
                    token_probs_clean.append(prob)
                    log_prob_sum += np.log(prob + 1e-10)

                    # Extend sequence for next token (if not last)
                    if i < len(target_tokens) - 1:
                        new_token = torch.tensor([[target_token_id]], device=current_inp['input_ids'].device)
                        current_inp['input_ids'] = torch.cat([current_inp['input_ids'], new_token], dim=1)
                        if 'attention_mask' in current_inp:
                            new_mask = torch.ones((1, 1), device=current_inp['attention_mask'].device)
                            current_inp['attention_mask'] = torch.cat([current_inp['attention_mask'], new_mask], dim=1)

            base_score = np.exp(log_prob_sum)
            answer = ''.join(decode_tokens(mt.tokenizer, target_tokens))
        else:
            # Single token case
            answer_t = target_tokens
            with torch.no_grad():
                outputs = mt(**inp)
                base_score = torch.softmax(outputs["logits"][0, -1, :], dim=0)[answer_t].item()
            [answer] = decode_tokens(mt.tokenizer, [answer_t])
            token_probs_clean = None
    else:
        # Use predicted token
        [answer] = decode_tokens(mt.tokenizer, [answer_t])
        target_tokens = answer_t
        token_probs_clean = None

    # Probability under corruption/substitution with no restoration
    low_score = trace_with_patch(
        mt.model,
        inp,
        [],  # no states restored
        target_tokens,
        tokens_to_mix,
        noise=noise,
        token_substitutions=token_substitutions,
        tokenizer=mt.tokenizer,
    )
    if torch.is_tensor(low_score):
        low_score = low_score.item()

    # Full importance sweep
    if not kind:
        differences = trace_important_states(
            mt.model,
            mt.num_layers,
            inp,
            tokens_to_mix,
            target_tokens,
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
            target_tokens,
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

    result = dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=subject_range,
        answer=answer,
        window=window,
        kind=kind or "",
        is_multitoken=is_multitoken,
    )

    # Add token-level probabilities for multi-token targets
    if is_multitoken and token_probs_clean is not None:
        result['token_probs'] = token_probs_clean

    return result


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
    plt.colorbar(label='Avg restoration (prob)')
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
        target_tokens=None,
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
                target_tokens=target_tokens,
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
                target_tokens=target_tokens,
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
                target_tokens=target_tokens,
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
        _save_per_layer_line(avg_hidden, "Average Hidden Flow “ Per Layer%s" % model_tag,
                             save_dir / "avg_hidden_per_layer.png")
    if avg_mlp is not None:
        _save_per_layer_line(avg_mlp, "Average MLP Flow “ Per Layer%s" % model_tag,
                             save_dir / "avg_mlp_per_layer.png")
    if avg_attn is not None:
        _save_per_layer_line(avg_attn, "Average Attention Flow “ Per Layer%s" % model_tag,
                             save_dir / "avg_attn_per_layer.png")

    return {"hidden": avg_hidden, "mlp": avg_mlp, "attn": avg_attn}


def trace_important_states(model, num_layers, inp, e_range, answer_t, noise=0.1,
                           token_substitutions=None, tokenizer=None):
    """
    Trace important states across all token positions and layers.
    Now supports multi-token targets (answer_t can be int or list).

    Args:
        model: The model
        num_layers: Number of layers in the model
        inp: Input dictionary
        e_range: Token range to corrupt (for noise) or None (for token substitution)
        answer_t: Answer token ID (int) or list of token IDs (for multi-token)
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
    Now supports multi-token targets (answer_t can be int or list).

    Args:
        model: The model
        num_layers: Number of layers in the model
        inp: Input dictionary
        e_range: Token range to corrupt (for noise) or None (for token substitution)
        answer_t: Answer token ID (int) or list of token IDs (for multi-token)
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
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    differences = result["scores"]

    # Debug prints (remove once confirmed)
    print("scores shape:", np.array(result["scores"]).shape)
    print("type(input_tokens):", type(result["input_tokens"]))
    print("len(input_tokens):", len(result["input_tokens"]))
    print("first 5 input_tokens:",
          result["input_tokens"][:5] if hasattr(result["input_tokens"], "__getitem__") else result["input_tokens"])

    low_score = result["low_score"]
    high_score = result["high_score"]
    answer = result["answer"]
    kind = result.get("kind", None)
    if kind == "":
        kind = None

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
            cmap={None: "Purples", "mlp": "Greens", "attn": "Reds"}[kind],
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

        # ✅ Safe placement: use axes-fraction coordinates (prevents bbox_inches='tight' blowups)
        cb.ax.text(
            1.05, 1.00, f"High score: {high_score:.3f}",
            transform=cb.ax.transAxes,
            fontsize=7, va="bottom", ha="left"
        )
        cb.ax.text(
            1.05, 0.00, f"Low score: {low_score:.3f}",
            transform=cb.ax.transAxes,
            fontsize=7, va="top", ha="left"
        )

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
        target_tokens=None,
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
        token_substitutions=token_substitutions, target_tokens=target_tokens
    )
    plot_trace_heatmap_with_scores(result, savepdf, modelname=modelname)


def trace_attention_heads(
        model, num_layers, num_heads, inp, e_range, answer_t, noise=0.1,
        token_substitutions=None, tokenizer=None
):
    """
    Trace the marginal importance of individual attention heads using
    ablation patching (patch-all-but-one).

    For each (position, layer, head) triple:
      1. Compute full_layer_score: patch ALL heads in this layer at this
         position from the clean run into the corrupted run.
      2. Compute leave_one_out_score: patch all heads EXCEPT the target head
         (i.e. zero the target head's contribution in hook_z on the corrupted
         run while patching the rest from clean).
      3. Return marginal_delta = full_layer_score - leave_one_out_score.

    A large positive marginal_delta means removing that head from the patch
    significantly hurts restoration — i.e. that head carries critical clean
    information.  This is robust to the superlinearity problem of
    patch-one-head approaches, because the other heads already supply clean
    context.

    Args:
        model:               The HookedTransformer model (or wrapper with .model).
        num_layers:          Number of transformer layers.
        num_heads:           Number of attention heads per layer.
        inp:                 Input dictionary (output of make_inputs).
        e_range:             Token range to corrupt with noise, or None when
                             token_substitutions is used.
        answer_t:            Answer token ID (int) or list of token IDs.
        noise:               Noise level (ignored when token_substitutions given).
        token_substitutions: List of (position, new_token_id) tuples, or None.
        tokenizer:           Tokenizer instance (used for debug printing only).

    Returns:
        Tensor of shape [num_positions, num_layers, num_heads] containing the
        marginal Δp for each (position, layer, head) triple.
        marginal_delta[pos, layer, head] = full_layer_score - leave_one_out_score
    """
    from transformer_lens.utilities import nethook as _nethook

    raw_model = model.model if hasattr(model, 'model') else model

    ntoks = inp["input_ids"].shape[1]
    table = []

    for tnum in range(ntoks):
        layer_results = []
        for layer in range(num_layers):

            # hook_z shape: [batch, seq, n_heads, d_head]
            head_hook_name = f"blocks.{layer}.attn.hook_z"

            # ── Step 1: full-layer patch score for this (tnum, layer) ─────────
            # No outer hook — all heads are patched from clean.
            full_layer_score = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer, 'attn'))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                token_substitutions=token_substitutions,
                tokenizer=tokenizer,
            )

            # ── Step 2: leave-one-out score for each head ─────────────────────
            head_results = []
            for head in range(num_heads):

                def make_ablator(ablated_head):
                    """
                    Hook that zeroes the ablated_head's contribution in hook_z
                    on the corrupted run (x[1:]), leaving x[0] (the clean run)
                    untouched so trace_with_patch can still read clean activations.
                    All other heads in x[1:] are left as-is (they will be
                    overwritten by trace_with_patch's patch_rep with clean values).
                    """
                    def ablate_head(x, layer_name):
                        # Zero only the target head on corrupted rows.
                        x[1:, :, ablated_head, :] = 0.0
                        return x
                    return ablate_head

                ablator = make_ablator(head)

                with _nethook.TraceDict(
                    raw_model,
                    [head_hook_name],
                    edit_output=ablator,
                ):
                    leave_one_out_score = trace_with_patch(
                        model,
                        inp,
                        [(tnum, layername(model, layer, 'attn'))],
                        answer_t,
                        tokens_to_mix=e_range,
                        noise=noise,
                        token_substitutions=token_substitutions,
                        tokenizer=tokenizer,
                    )

                # Marginal contribution: how much does this head add to the
                # full-layer restoration?
                marginal_delta = full_layer_score - leave_one_out_score
                head_results.append(marginal_delta)

            layer_results.append(torch.stack(head_results))
        table.append(torch.stack(layer_results))

    return torch.stack(table)  # [num_positions, num_layers, num_heads]


def calculate_attention_head_flow(
        mt, prompt, subject, samples=10, noise=0.1,
        token_substitutions=None, target_tokens=None
):
    """
    Calculate attention head ablation patching scores across all token positions.

    For each (position, layer, head) triple, computes the marginal contribution
    of that head using patch-all-but-one ablation:
      marginal_delta = full_layer_score - leave_one_out_score
    where full_layer_score patches all heads from clean and leave_one_out_score
    patches all heads except the target from clean (zeroing the target head).

    A large positive score means the head carries critical clean information.
    Scores are already in Δp units and do not require further subtraction of
    low_score at plot time.

    Args:
        mt:                  Model wrapper with .model, .num_layers, .tokenizer.
        prompt:              Clean input prompt string.
        subject:             Subject span for noise-based corruption, or None
                             when token_substitutions is used.
        samples:             Number of noise samples (ignored for token substitution).
        noise:               Noise level (ignored for token substitution).
        token_substitutions: List of (position, new_token_id) tuples, or None.
        target_tokens:       Single token ID (int), list of token IDs for a
                             multi-token target, or None to use the model's
                             top predicted token.

    Returns:
        dict with keys:
            scores:        [num_positions, num_layers, num_heads] tensor of
                           marginal Δp values (full_layer - leave_one_out).
            low_score:     float — p(answer) under full corruption, no patching.
            high_score:    float — p(answer) under clean input.
            input_ids:     token ID tensor for the clean prompt.
            input_tokens:  list of decoded token strings.
            subject_range: (start, end) tuple of substituted/corrupted positions.
            answer:        decoded string of the target token(s).
    """
    # ── 1. Determine number of attention heads ────────────────────────────────
    if hasattr(mt.model, 'cfg') and hasattr(mt.model.cfg, 'n_heads'):
        num_heads = mt.model.cfg.n_heads
    elif hasattr(mt.model, 'config') and hasattr(mt.model.config, 'n_head'):
        num_heads = mt.model.config.n_head
    else:
        num_heads = 12  # fallback

    # ── 2. Build inputs ───────────────────────────────────────────────────────
    if token_substitutions is not None:
        inp = make_inputs(mt.tokenizer, [prompt])
        e_range = None
        tokens_to_mix = None
    else:
        inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
        e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
        tokens_to_mix = e_range

    # ── 3. Resolve target and compute high_score (clean probability) ──────────
    is_multitoken = isinstance(target_tokens, (list, tuple))

    # Initial prediction to get answer_t if target_tokens is not supplied.
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt, inp)]

    if target_tokens is not None:
        if is_multitoken:
            # Joint probability over the full multi-token target sequence,
            # autoregressive: p(t0) * p(t1|t0) * ...
            log_prob_sum = 0.0
            current_inp = {
                k: v[0:1].clone() if torch.is_tensor(v) else v
                for k, v in inp.items()
            }
            with torch.no_grad():
                for i, target_token_id in enumerate(target_tokens):
                    outputs = mt(**current_inp)
                    probs = torch.softmax(outputs["logits"][0, -1, :], dim=0)
                    log_prob_sum += np.log(probs[target_token_id].item() + 1e-10)

                    if i < len(target_tokens) - 1:
                        new_token = torch.tensor(
                            [[target_token_id]],
                            device=current_inp['input_ids'].device
                        )
                        current_inp['input_ids'] = torch.cat(
                            [current_inp['input_ids'], new_token], dim=1
                        )
                        if 'attention_mask' in current_inp:
                            new_mask = torch.ones(
                                (1, 1), device=current_inp['attention_mask'].device
                            )
                            current_inp['attention_mask'] = torch.cat(
                                [current_inp['attention_mask'], new_mask], dim=1
                            )

            base_score = np.exp(log_prob_sum)
            answer_t = target_tokens
            # Decode all target tokens and join for display.
            answer = ''.join(decode_tokens(mt.tokenizer, target_tokens))

        else:
            # Single token override.
            answer_t = target_tokens
            with torch.no_grad():
                outputs = mt(**inp)
                base_score = torch.softmax(
                    outputs["logits"][0, -1, :], dim=0
                )[answer_t].item()
            [answer] = decode_tokens(mt.tokenizer, [answer_t])

    else:
        # Use the model's own top prediction.
        [answer] = decode_tokens(mt.tokenizer, [answer_t])

    # ── 4. Low score — p(answer) with full corruption and no patching ─────────
    low_score = trace_with_patch(
        mt.model,
        inp,
        [],
        answer_t,
        tokens_to_mix,
        noise=noise,
        token_substitutions=token_substitutions,
        tokenizer=mt.tokenizer,
    )
    if torch.is_tensor(low_score):
        low_score = low_score.item()

    # ── 5. Per-head activation patching sweep ─────────────────────────────────
    differences = trace_attention_heads(
        mt.model,
        mt.num_layers,
        num_heads,
        inp,
        tokens_to_mix,        # e_range for noise, None for token substitution
        answer_t,
        noise=noise,
        token_substitutions=token_substitutions,
        tokenizer=mt.tokenizer,
    )
    differences = differences.detach().cpu()

    # ── 6. Subject / substitution range for the plot ──────────────────────────
    if token_substitutions is None:
        subject_range = e_range
    else:
        if token_substitutions:
            positions = sorted(pos for pos, _ in token_substitutions)
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
                                        title="Attention Head Activation Patching (All Pos)",
                                        top_k=10):
    """
    Plot average attention head ablation patching results across all positions.
    Only highlights the top-k heads with highest absolute marginal effect.

    Args:
        results_list: List of result dicts from calculate_attention_head_flow.
                      Each dict must contain:
                        scores: [num_positions, num_layers, num_heads] tensor
                                of marginal Δp values (already in Δp units;
                                no low_score subtraction needed).
        savepdf: Path to save PDF, or None to display interactively.
        title:   Plot title.
        top_k:   Number of top heads to highlight by |marginal Δp| (default: 10).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # ── 1. Average marginal Δp scores over positions and across prompts ───────
    # scores from trace_attention_heads are already marginal Δp values:
    #   marginal_delta = full_layer_score - leave_one_out_score
    # No further subtraction of low_score is needed.
    all_delta_scores = []
    for result in results_list:
        scores = result["scores"]   # [num_positions, num_layers, num_heads]
        # Average over token positions → [num_layers, num_heads]
        avg_over_pos = scores.mean(dim=0)
        all_delta_scores.append(avg_over_pos)

    # Stack and average across prompts → [num_layers, num_heads]
    stacked = torch.stack(all_delta_scores)
    avg_scores = stacked.mean(dim=0).numpy()  # Δp averaged over positions & prompts

    # ── 2. Top-k masking ──────────────────────────────────────────────────────
    abs_scores = np.abs(avg_scores)
    flat_abs_scores = abs_scores.flatten()
    threshold = np.sort(flat_abs_scores)[-top_k] if len(flat_abs_scores) >= top_k else 0

    # NaN out non-top-k cells for display; keep avg_scores intact for printing.
    masked_scores = np.where(abs_scores >= threshold, avg_scores, np.nan)

    # ── 3. Colormap scaling from full (unmasked) Δp range ────────────────────
    # Use avg_scores (not masked_scores) so that vmax reflects the true score
    # distribution and mid-range heads are not artificially washed out.
    vmax = np.abs(avg_scores).max()
    if vmax == 0:
        vmax = 1e-6  # guard against all-zero edge case

    # ── 4. Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)

    h = ax.imshow(
        masked_scores,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
        origin="lower",
    )
    h.cmap.set_bad(color="lightgray", alpha=0.3)

    num_layers, num_heads = avg_scores.shape
    ax.set_xlabel("Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(f"{title}\n(Top {top_k} heads by |marginal Δp|)", fontsize=11)

    ax.set_xticks(range(num_heads))
    ax.set_yticks(range(num_layers))

    # Minor ticks for grid lines
    ax.set_xticks(np.arange(-0.5, num_heads, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_layers, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)

    cb = plt.colorbar(h, ax=ax)
    cb.set_label("Marginal Δp (full layer − leave-one-out)", fontsize=11)

    plt.tight_layout()

    if savepdf:
        Path(savepdf).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # ── 5. Console summary ────────────────────────────────────────────────────
    print(f"\nTop {top_k} attention heads by absolute marginal Δp (ablation patching):")
    flat_indices = np.argsort(abs_scores.flatten())[-top_k:][::-1]
    for rank, idx in enumerate(flat_indices, 1):
        layer = idx // num_heads
        head = idx % num_heads
        effect = avg_scores[layer, head]
        print(f"  {rank}. Layer {layer}, Head {head}: marginal Δp = {effect:.6f}")

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
        target_tokens=None,
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
            target_tokens=target_tokens,
        )


def _save_per_layer_line(avg_matrix: np.ndarray, title: str, outfile: Path):
    """
    Plot mean importance per layer (averaged over token positions) and save.
    """
    if avg_matrix is None:
        return
    # mean over token dimension (axis=1 â†’ per-layer)
    y = np.nanmean(avg_matrix, axis=1)
    x = np.arange(len(y))

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker='o')
    plt.xlabel('Layer')
    plt.ylabel('Mean restoration Î” (prob)')
    plt.title(title)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=200)
    plt.close()


@hydra.main(config_path="causal_transformers/config", config_name="config", version_base=None)
def main(cfg):
    # ── Model loading ────────────────────────────────────────────────────────
    checkpoint_path = "causal_transformers/checkpoints/gaussian/gaussian_dataset_v6/hooked_transformer_deep/instance3/checkpoint_0300.pth"
    print(f"Loading fine-tuned model from: {checkpoint_path}")
    model = load_finetuned_model(checkpoint_path, cfg)
    print("MODEL TYPE:", type(model))
    print("HAS transformer:", hasattr(model, "transformer"))
    print("HAS h:", hasattr(model, "h"))
    print("HAS module wrapper:", hasattr(model, "module"))
    if hasattr(model, "module"):
        print("  UNDERLYING HAS transformer:", hasattr(model.module, "transformer"))
        print("  UNDERLYING HAS h:", hasattr(model.module, "h"))
    if hasattr(model, "transformer"):
        print("TRANSFORMER TYPE:", type(model.transformer))
        print("TRANSFORMER HAS h:", hasattr(model.transformer, "h"))
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

    # ── SCM prompt file ──────────────────────────────────────────────────────
    # After running predict_and_append.py, each line has the form:
    #   INFERENCE <scm_idx> [OBS Vi v]* [DO Vi v]* <query_var> <predicted_value>
    # Pairs are interleaved: clean line then corrupt line.
    # The prompt passed to patching functions is everything except the last
    # token (the predicted value); the last token IS the target.
    # We use the intervention corruption type here.
    template_path = "causal_transformers/inference/variable_value.txt"
    corruption_type = "variable_value"

    pairs = load_template_pairs(template_path)[:100]
    outdir = Path(f"scm_ablation_patch_{corruption_type}")
    outdir.mkdir(parents=True, exist_ok=True)

    attention_head_results = []

    for idx, (clean_full, corrupted_full) in enumerate(tqdm(pairs, desc="Pairs")):
        clean_words = clean_full.split()
        corrupted_words = corrupted_full.split()

        # The last token is the model's predicted mean value (appended by
        # predict_and_append.py).  Everything before it is the prompt.
        if len(clean_words) < 2:
            print(f"[warn] Skipping #{idx:03d}: line too short: '{clean_full}'")
            continue

        predicted_token_str = clean_words[-1]          # e.g. '-4.0' or '2.3'
        clean_prompt        = " ".join(clean_words[:-1])
        corrupted_prompt    = " ".join(corrupted_words[:-1])

                # Resolve the predicted token to its ID.
        # With the Preprocessor, each whitespace-separated word is a single atomic token,
        # so numeric values like "-4.0" are single vocab entries.
        if not hasattr(mt.tokenizer, "vocab") or predicted_token_str not in mt.tokenizer.vocab:
            print(f"[warn] Skipping #{idx:03d}: '{predicted_token_str}' not in tokenizer vocab")
            continue
        target_tokens = mt.tokenizer.vocab[predicted_token_str]

        # ── Build token substitutions (clean → corrupt) ────────────────────── (clean → corrupt) ──────────────────────
        try:
            substitutions = find_word_token_substitutions(
                mt.tokenizer, clean_prompt, corrupted_prompt
            )
        except ValueError as e:
            print(f"[warn] Skipping #{idx:03d}: {e}")
            continue

        # ── Save prefix for plots ─────────────────────────────────────────────
        short = re.sub(r"\W+", "_", clean_prompt.lower()).strip("_")
        short = (short[:60] + "…") if len(short) > 60 else short
        save_prefix = str(outdir / f"{idx:03d}_{short}")


        # ── Hidden-flow plots (all / mlp / attn) ─────────────────────────────
        plot_all_flow(
            mt,
            clean_prompt,
            token_substitutions=substitutions,
            modelname="Fine-tuned GPT-2 (SCM)",
            save_prefix=save_prefix,
            target_tokens=target_tokens,
        )

        # ── Attention-head flow ───────────────────────────────────────────────
        try:
            head_result = calculate_attention_head_flow(
                mt,
                clean_prompt,
                subject=None,
                token_substitutions=substitutions,
                target_tokens=target_tokens,
            )
            attention_head_results.append(head_result)
        except Exception as e:
            print(f"[warn] Attention head flow failed on #{idx:03d}: {e}")

    # ── Aggregate attention-head plot ────────────────────────────────────────
    if attention_head_results:
        plot_attention_head_heatmap_average(
            attention_head_results,
            savepdf=str(outdir / "avg_attention_heads.pdf"),
            title=f"Avg Attention Head Flow — {corruption_type} corruption",
        )

    print(f"\nDone! Plots saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()

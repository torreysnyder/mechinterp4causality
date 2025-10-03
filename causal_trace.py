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
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
    ):
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model is None:
            assert model_name is not None
            model = AutoModelForCausalLM.from_pretrained(
                model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
            )
            nethook.set_requires_grad(False, model)
            model.eval().cuda()
        self.tokenizer = tokenizer
        self.model = model
        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )

def layername(model, num, kind=None):
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    assert False, "unknown transformer structure"

def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()

def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
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
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"

    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                kind
            ],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )

def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_loc = whole_string.index(substring)
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
    Predict tokens from prompts.

    Args:
        mt: ModelAndTokenizer instance
        prompts: List of prompt strings
        return_p: If True, return probabilities along with predictions
        return_logits_for: If provided, return logits for specified token(s).
                          Can be:
                          - A string (e.g., "Paris")
                          - A list of strings (e.g., ["Paris", "London"])
                          - A token ID (int)
                          - A list of token IDs

    Returns:
        result: List of decoded predictions
        p: (optional) Probabilities if return_p=True
        logits: (optional) Logits for specified token(s) if return_logits_for is provided
    """
    inp = make_inputs(mt.tokenizer, prompts)
    preds, p, out = predict_from_input(mt.model, inp, return_logits=True)
    result = [mt.tokenizer.decode(c) for c in preds]

    returns = [result]

    if return_p:
        returns.append(p)

    if return_logits_for is not None:
        # Convert token strings to IDs if needed
        if isinstance(return_logits_for, str):
            # Single token string
            token_ids = mt.tokenizer.encode(return_logits_for, add_special_tokens=False)
            if len(token_ids) == 0:
                raise ValueError(f"Token '{return_logits_for}' could not be encoded")
            if len(token_ids) > 1:
                print(
                    f"Note: '{return_logits_for}' encodes to {len(token_ids)} tokens: {token_ids}. Returning probabilities for all.")
            # Use all token IDs, not just the first one
            token_id = token_ids  # This is now a list if multiple tokens
        elif isinstance(return_logits_for, (list, tuple)):
            # Check if it's a list of strings or IDs
            if all(isinstance(x, str) for x in return_logits_for):
                # List of token strings
                token_ids = []
                for token_str in return_logits_for:
                    tids = mt.tokenizer.encode(token_str, add_special_tokens=False)
                    if len(tids) == 0:
                        raise ValueError(f"Token '{token_str}' could not be encoded")
                    if len(tids) > 1:
                        print(f"Note: '{token_str}' encodes to {len(tids)} tokens: {tids}. Including all.")
                    # Add all token IDs from this string
                    token_ids.extend(tids)
                token_id = token_ids
            else:
                # List of token IDs
                token_id = return_logits_for
        else:
            # Single token ID
            token_id = return_logits_for

        # Convert logits to probabilities using softmax
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
    Predict from input tensors.

    Args:
        model: The language model
        inp: Input dictionary with input_ids and attention_mask
        return_logits: If True, return full logits tensor

    Returns:
        preds: Predicted token IDs
        p: Probabilities of predictions
        out: (optional) Full logits tensor if return_logits=True
    """
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)

    if return_logits:
        return preds, p, out
    return preds, p

def collect_embedding_std(mt, subjects):
    alldata = []
    for s in subjects:
        inp = make_inputs(mt.tokenizer, [s])
        with nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
            mt.model(**inp)
            alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()
    return noise_level

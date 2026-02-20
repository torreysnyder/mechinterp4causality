import os
import torch, numpy
from collections import defaultdict

from causal_trace import (
    ModelAndTokenizer,
    make_inputs,
    decode_tokens,
)

torch.set_grad_enabled(False)

model_name = "gpt2"
mt = ModelAndTokenizer(
    model_name,
    torch_dtype=torch.float16,
)


def analyze_conditional_attention(
        model,
        tokenizer,
        sentences,
        delimiter_tokens=["if", "then"]
):
    """
    Analyze attention patterns for conditional delimiters like "if" and "then".

    Args:
        model: The transformer model
        tokenizer: The tokenizer
        sentences: List of conditional sentences to analyze
        delimiter_tokens: List of delimiter words to analyze (default: ["if", "then"])

    Returns:
        Dictionary with attention statistics per layer and head
    """
    results = {delim: defaultdict(lambda: defaultdict(list))
               for delim in delimiter_tokens}

    for sentence in sentences:
        # Tokenize the sentence
        inp = make_inputs(tokenizer, [sentence])
        tokens = decode_tokens(tokenizer, inp["input_ids"][0])

        # Find delimiter positions
        delimiter_positions = {}
        for delim in delimiter_tokens:
            # Look for the delimiter in the tokens
            for i, token in enumerate(tokens):
                if delim in token.lower().strip():
                    delimiter_positions[delim] = i
                    break

        if not delimiter_positions:
            continue

        # Run model and capture attention patterns
        with torch.no_grad():
            outputs = model(**inp, output_attentions=True)
            attentions = outputs.attentions  # Tuple of attention tensors per layer

        # Analyze attention for each layer and head
        num_layers = len(attentions)
        for layer_idx in range(num_layers):
            attention = attentions[layer_idx][0]  # [num_heads, seq_len, seq_len]
            num_heads = attention.shape[0]

            for head_idx in range(num_heads):
                head_attention = attention[head_idx]  # [seq_len, seq_len]

                # Calculate proportion of attention to each delimiter
                for delim, pos in delimiter_positions.items():
                    # P_d: proportion of attention paid TO the delimiter
                    attention_to_delim = head_attention[:, pos].sum().item()
                    total_attention = head_attention.sum().item()

                    if total_attention > 0:
                        proportion = attention_to_delim / total_attention
                        results[delim][layer_idx][head_idx].append(proportion)

    # Average across all sentences
    averaged_results = {}
    for delim in delimiter_tokens:
        averaged_results[delim] = numpy.zeros((len(results[delim]),
                                               max(len(results[delim][l])
                                                   for l in results[delim])))
        for layer_idx in results[delim]:
            for head_idx in results[delim][layer_idx]:
                if results[delim][layer_idx][head_idx]:
                    avg = numpy.mean(results[delim][layer_idx][head_idx])
                    averaged_results[delim][layer_idx, head_idx] = avg

    return averaged_results


def find_conditional_ranges(tokenizer, sentence):
    """
    Automatically detect condition and effect clause ranges in a conditional sentence.

    Args:
        tokenizer: The tokenizer
        sentence: A conditional sentence (e.g., "If X, then Y")

    Returns:
        Tuple of (condition_range, effect_range) or None if delimiters not found
        where each range is (start_idx, end_idx)
    """
    # Tokenize the sentence
    tokens = tokenizer.encode(sentence)
    token_strs = [tokenizer.decode([t]).lower().strip() for t in tokens]

    # Find positions of "if" and "then"
    if_pos = None
    then_pos = None

    for i, token in enumerate(token_strs):
        if 'if' in token and if_pos is None:
            if_pos = i
        if 'then' in token and then_pos is None:
            then_pos = i

    # If we can't find both delimiters, return None
    if if_pos is None or then_pos is None:
        return None

    # Condition range: from after "if" to before "then"
    condition_start = if_pos + 1
    condition_end = then_pos

    # Effect range: from after "then" to end of sentence
    effect_start = then_pos + 1
    effect_end = len(tokens)

    return ((condition_start, condition_end), (effect_start, effect_end))


def analyze_conditional_causal_attention(
        model,
        tokenizer,
        sentences
):
    """
    Analyze condition-to-effect attention patterns with automatic range detection.

    Args:
        model: The transformer model
        tokenizer: The tokenizer
        sentences: List of conditional sentences

    Returns:
        Dictionary with causal attention statistics per layer and head
    """
    results = defaultdict(lambda: defaultdict(list))
    skipped = 0

    for sentence in sentences:
        # Automatically detect condition and effect ranges
        ranges = find_conditional_ranges(tokenizer, sentence)

        if ranges is None:
            print(f"Warning: Could not find 'if' and 'then' in: {sentence[:50]}...")
            skipped += 1
            continue

        (cond_start, cond_end), (effect_start, effect_end) = ranges

        inp = make_inputs(tokenizer, [sentence])

        with torch.no_grad():
            outputs = model(**inp, output_attentions=True)
            attentions = outputs.attentions

        num_layers = len(attentions)
        for layer_idx in range(num_layers):
            attention = attentions[layer_idx][0]  # [num_heads, seq_len, seq_len]
            num_heads = attention.shape[0]

            for head_idx in range(num_heads):
                head_attention = attention[head_idx]

                # Calculate P_c: proportion of condition-to-effect attention
                # Attention from effect tokens to condition tokens
                if effect_start < effect_end and cond_start < cond_end:
                    causal_attention = head_attention[effect_start:effect_end,
                                       cond_start:cond_end].sum().item()
                    total_attention = head_attention.sum().item()

                    if total_attention > 0:
                        proportion = causal_attention / total_attention
                        results[layer_idx][head_idx].append(proportion)

    if skipped > 0:
        print(f"Skipped {skipped} sentences due to missing delimiters.")

    # Average across all sentences
    num_layers = len(results)
    num_heads = max(len(results[l]) for l in results) if results else 12
    averaged_results = numpy.zeros((num_layers, num_heads))

    for layer_idx in results:
        for head_idx in results[layer_idx]:
            if results[layer_idx][head_idx]:
                averaged_results[layer_idx, head_idx] = numpy.mean(
                    results[layer_idx][head_idx]
                )

    return averaged_results


def plot_delimiter_attention_heatmap(results, delimiter, savepdf=None):
    """
    Plot heatmap showing attention paid to a specific delimiter.

    Args:
        results: 2D numpy array [num_layers, num_heads]
        delimiter: Name of the delimiter (e.g., "if", "then")
        savepdf: Optional path to save the plot
    """
    import matplotlib.pyplot as plt

    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

        im = ax.imshow(results, cmap='Blues', aspect='auto')

        ax.set_xlabel('Head', fontsize=12)
        ax.set_ylabel('Layer', fontsize=12)
        ax.set_title(f'Proportion of Attention Paid to "{delimiter}"', fontsize=14)

        # Set ticks
        num_layers, num_heads = results.shape
        ax.set_yticks(range(num_layers))
        ax.set_yticklabels(range(num_layers))
        ax.set_xticks(range(num_heads))
        ax.set_xticklabels(range(num_heads))

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Proportion', fontsize=10)

        plt.tight_layout()

        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


# Load conditional sentences from file
def load_conditional_sentences(filepath):
    """
    Load conditional sentences from a text file.

    Args:
        filepath: Path to the text file containing sentences (one per line)

    Returns:
        List of sentence strings
    """
    sentences = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    sentences.append(line)
        print(f"Loaded {len(sentences)} sentences from {filepath}")
        return sentences
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        print("Using default example sentences instead.")
        # Fallback to example sentences
        return [
            "If it rains tomorrow, then I will stay home.",
            "If you study hard, then you will pass the exam.",
            "If the door is open, then we can enter the building.",
            "If she arrives early, then we can start the meeting.",
            "If he finishes his work, then he will join us for dinner.",
            "If the weather is nice, then we will go to the park.",
            "If you press the button, then the light will turn on.",
            "If the train is late, then we will miss our connection.",
            "If I have time, then I will call you later.",
            "If they win the game, then they will advance to finals.",
        ]


# Load sentences from templates.txt
conditional_sentences = load_conditional_sentences("templates.txt")

# Analyze attention to delimiters
print("Analyzing attention to conditional delimiters...")
delimiter_results = analyze_conditional_attention(
    mt.model,
    mt.tokenizer,
    conditional_sentences,
    delimiter_tokens=["if", "then"]
)

# Plot results for each delimiter
for delimiter, results in delimiter_results.items():
    print(f"\nPlotting attention heatmap for '{delimiter}'...")
    plot_delimiter_attention_heatmap(
        results,
        delimiter,
        savepdf=f"results/conditional_attention_{delimiter}.pdf"
    )

# Analyze condition-to-effect attention patterns
print("\nAnalyzing condition-to-effect attention patterns...")

causal_attention_results = analyze_conditional_causal_attention(
    mt.model,
    mt.tokenizer,
    conditional_sentences
)

print("\nPlotting condition-to-effect attention heatmap...")
plot_delimiter_attention_heatmap(
    causal_attention_results,
    "Condition-to-Effect",
    savepdf="results/conditional_causal_attention.pdf"
)

print("\nAnalysis complete! Results saved to 'results/' directory.")

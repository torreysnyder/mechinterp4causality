import torch
import csv
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path

# Load GPT-2 model and tokenizer
print("Loading GPT-2 model and tokenizer...")
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()


def get_last_word_likelihood(sentence, model, tokenizer):
    """
    Calculate the likelihood of the last word in a sentence given the context.

    Args:
        sentence: Full sentence
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer

    Returns:
        Probability of the last word given the preceding context
    """
    # Split sentence into words
    words = sentence.strip().split()
    if len(words) < 2:
        return None

    # Get the last word and the context (everything before it)
    last_word = words[-1]
    context = ' '.join(words[:-1])

    # Tokenize context and full sentence
    context_ids = tokenizer.encode(context, return_tensors='pt')
    full_ids = tokenizer.encode(sentence, return_tensors='pt')

    # Get the token IDs for the last word
    # These are the tokens that appear in full_ids but not in context_ids
    last_word_start_idx = context_ids.shape[1]
    last_word_token_ids = full_ids[0, last_word_start_idx:]

    if len(last_word_token_ids) == 0:
        return None

    # Calculate probability for each token in the last word
    with torch.no_grad():
        # Get logits for the full sequence
        outputs = model(full_ids)
        logits = outputs.logits

        # Calculate probability for each token of the last word
        total_log_prob = 0.0
        for i, token_id in enumerate(last_word_token_ids):
            # Position in the sequence where this token appears
            pos = last_word_start_idx + i

            # Get logits at the previous position (to predict this token)
            token_logits = logits[0, pos - 1, :]

            # Convert to probabilities
            probs = torch.softmax(token_logits, dim=-1)

            # Get probability of the actual token
            token_prob = probs[token_id].item()

            # Add log probability
            total_log_prob += torch.log(torch.tensor(token_prob)).item()

        # Convert back to probability (geometric mean for multi-token words)
        avg_log_prob = total_log_prob / len(last_word_token_ids)
        final_prob = torch.exp(torch.tensor(avg_log_prob)).item()

    return final_prob


# Load the text file
file_path = 'causal_disjunction_templates.txt'
print(f"Loading file: {file_path}")

with open(file_path, 'r') as f:
    sentences = [line.strip() for line in f.readlines() if line.strip()]

# Process sentence pairs
results = []
print(f"\nProcessing {len(sentences)} sentences ({len(sentences) // 2} pairs)...")

for i in range(0, len(sentences), 2):
    if i + 1 >= len(sentences):
        break

    sentence1 = sentences[i]
    sentence2 = sentences[i + 1]

    print(f"\nPair {i // 2 + 1}/{len(sentences) // 2}")
    print(f"  Sentence 1: {sentence1}")
    print(f"  Sentence 2: {sentence2}")

    # Get likelihoods
    likelihood1 = get_last_word_likelihood(sentence1, model, tokenizer)
    likelihood2 = get_last_word_likelihood(sentence2, model, tokenizer)

    print(f"  Likelihood 1: {likelihood1:.6f}" if likelihood1 else "  Likelihood 1: None")
    print(f"  Likelihood 2: {likelihood2:.6f}" if likelihood2 else "  Likelihood 2: None")

    results.append({
        'sentence1': sentence1,
        'sentence2': sentence2,
        'likelihood1': likelihood1,
        'likelihood2': likelihood2
    })

# Write results to CSV
output_file = 'sentence_pair_likelihoods.csv'
print(f"\nWriting results to {output_file}...")

with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)

    # Write header
    writer.writerow(['likelihood_sentence1', 'likelihood_sentence2'])

    # Write data
    for result in results:
        writer.writerow([
            result['likelihood1'] if result['likelihood1'] is not None else '',
            result['likelihood2'] if result['likelihood2'] is not None else ''
        ])

print(f"Done! Results saved to {output_file}")
print(f"Total pairs processed: {len(results)}")

# Print summary statistics
valid_results = [(r['likelihood1'], r['likelihood2'])
                 for r in results
                 if r['likelihood1'] is not None and r['likelihood2'] is not None]

if valid_results:
    import numpy as np

    likes1 = [r[0] for r in valid_results]
    likes2 = [r[1] for r in valid_results]

    print("\nSummary Statistics:")
    print(f"Sentence 1 - Mean: {np.mean(likes1):.6f}, Std: {np.std(likes1):.6f}")
    print(f"Sentence 2 - Mean: {np.mean(likes2):.6f}, Std: {np.std(likes2):.6f}")

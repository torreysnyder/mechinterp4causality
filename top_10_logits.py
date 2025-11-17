import os
import torch
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
print("Loading GPT-2 model and tokenizer...")
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()

# Create output directory
output_dir = 'top_10_logits_disjunction'
os.makedirs(output_dir, exist_ok=True)

# Load the text file
file_path = 'causal_disjunction_templates.txt'
print(f"Loading file: {file_path}")

with open(file_path, 'r') as f:
    sentences = f.readlines()

# Process each sentence
for idx, sentence in enumerate(sentences):
    sentence = sentence.strip()
    if not sentence:
        continue

    # Remove the last word from the sentence
    words = sentence.split()
    if len(words) < 2:
        continue

    truncated_sentence = ' '.join(words[:-1])
    print(f"\nProcessing sentence {idx + 1}/{len(sentences)}")
    print(f"Original: {sentence}")
    print(f"Truncated: {truncated_sentence}")

    # Tokenize the truncated sentence
    input_ids = tokenizer.encode(truncated_sentence, return_tensors='pt')

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Get logits for the next token (last position)
    next_token_logits = logits[0, -1, :]

    # Convert logits to probabilities
    probs = torch.softmax(next_token_logits, dim=-1)

    # Get top 10 tokens
    top_probs, top_indices = torch.topk(probs, 10)

    # Convert to lists
    top_probs = top_probs.cpu().numpy()
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices.cpu().numpy()]

    # Create bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(10), top_probs, color='steelblue', edgecolor='black')
    plt.xlabel('Token Rank', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'Top 10 Most Likely Next Tokens\n"{truncated_sentence}"', fontsize=10)
    plt.xticks(range(10), top_tokens, rotation=45, ha='right')
    plt.ylim(0, max(top_probs) * 1.1)

    # Add probability values on top of bars
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{prob:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, f'sentence_{idx + 1:03d}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved plot to: {output_path}")
    print(f"Top token: '{top_tokens[0]}' with probability {top_probs[0]:.4f}")

print(f"\nAll plots saved to directory: {output_dir}")
print(f"Total plots created: {len([f for f in os.listdir(output_dir) if f.endswith('.png')])}")

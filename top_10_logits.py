import os
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
print("Loading GPT-2 model and tokenizer...")
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()

# Create output directory
output_dir = 'top_100_cosine_similarity'
os.makedirs(output_dir, exist_ok=True)

# Load the text file
file_path = 'causal_disjunction_templates.txt'
print(f"Loading file: {file_path}")

with open(file_path, 'r') as f:
    sentences = f.readlines()

# Get the embedding matrix from the model
embedding_matrix = model.transformer.wte.weight  # Shape: [vocab_size, hidden_size]

# Process each sentence
for idx, sentence in enumerate(sentences):
    sentence = sentence.strip()
    if not sentence:
        continue

    # Split sentence into words
    words = sentence.split()
    if len(words) < 2:
        continue

    # Get the target word (last word)
    target_word = words[-1]
    truncated_sentence = ' '.join(words[:-1])

    print(f"\nProcessing sentence {idx + 1}/{len(sentences)}")
    print(f"Original: {sentence}")
    print(f"Truncated: {truncated_sentence}")
    print(f"Target word: {target_word}")

    # Tokenize the target word to get its token ID
    target_token_ids = tokenizer.encode(target_word, add_special_tokens=False)
    if len(target_token_ids) == 0:
        print(f"Warning: Could not tokenize target word '{target_word}', skipping...")
        continue

    target_token_id = target_token_ids[0]  # Use first token if word is split
    target_token_str = tokenizer.decode([target_token_id])

    # Get the embedding vector for the target token
    target_embedding = embedding_matrix[target_token_id]  # Shape: [hidden_size]

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

    # Get top 100 tokens by probability
    top_probs, top_indices = torch.topk(probs, 100)

    # Convert to CPU numpy arrays
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()

    # Get embeddings for top 100 tokens
    top_embeddings = embedding_matrix[top_indices]  # Shape: [100, hidden_size]

    # Calculate cosine similarity between each top token and the target token
    # Normalize vectors for cosine similarity
    target_embedding_norm = F.normalize(target_embedding.unsqueeze(0), dim=1)  # Shape: [1, hidden_size]
    top_embeddings_norm = F.normalize(top_embeddings, dim=1)  # Shape: [100, hidden_size]

    # Compute cosine similarities
    cosine_similarities = torch.mm(top_embeddings_norm, target_embedding_norm.t()).squeeze()  # Shape: [100]
    cosine_similarities = cosine_similarities.detach().cpu().numpy()

    # Create ranking data
    ranking_data = []
    for i in range(len(top_indices)):
        token_id = top_indices[i]
        token_str = tokenizer.decode([token_id])
        prob = top_probs[i]
        cosine_sim = cosine_similarities[i]
        ranking_data.append((token_str, cosine_sim, prob))

    # Sort by cosine similarity (descending)
    ranking_data.sort(key=lambda x: x[1], reverse=True)

    # Save to text file
    output_path = os.path.join(output_dir, f'sentence_{idx + 1:03d}_ranking.txt')
    with open(output_path, 'w') as f:
        f.write(f"Original sentence: {sentence}\n")
        f.write(f"Truncated sentence: {truncated_sentence}\n")
        f.write(f"Target token: '{target_token_str}'\n")
        f.write(f"\n{'=' * 80}\n")
        f.write(f"Top 100 tokens ranked by cosine similarity to target token\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"{'Rank':<6} {'Token':<20} {'Cosine Similarity':<20} {'Probability':<15}\n")
        f.write(f"{'-' * 80}\n")

        for rank, (token, cosine_sim, prob) in enumerate(ranking_data, 1):
            # Escape special characters for display
            token_display = repr(token)[1:-1]  # Remove outer quotes from repr
            f.write(f"{rank:<6} {token_display:<20} {cosine_sim:<20.6f} {prob:<15.6f}\n")

    print(f"Saved ranking to: {output_path}")
    print(
        f"Top token by cosine similarity: '{ranking_data[0][0]}' (similarity: {ranking_data[0][1]:.4f}, prob: {ranking_data[0][2]:.4f})")

print(f"\nAll rankings saved to directory: {output_dir}")
print(f"Total files created: {len([f for f in os.listdir(output_dir) if f.endswith('.txt')])}")

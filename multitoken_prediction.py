import torch


def calculate_multitoken_probability(model, tokenizer, inp, target_tokens, device=None):
    """
    Calculate the joint probability of a multi-token target word using conditional probabilities.

    Args:
        model: The language model
        tokenizer: The tokenizer
        inp: Input dictionary with 'input_ids' (batch_size=1)
        target_tokens: List of token IDs that make up the target word
        device: Device to use (if None, uses inp device)

    Returns:
        total_prob: Joint probability P(token₁, token₂, ..., tokenₙ)
        token_probs: List of individual conditional probabilities for each token
    """
    if device is None:
        device = inp['input_ids'].device

    # Start with the base input
    current_input_ids = inp['input_ids'].clone()

    token_probs = []
    log_prob_sum = 0.0

    with torch.no_grad():
        for i, target_token_id in enumerate(target_tokens):
            # Get model output for current sequence
            outputs = model(**{k: v for k, v in inp.items() if k in ['input_ids', 'attention_mask']})
            logits = outputs.logits

            # Get probability distribution for the next token (at last position)
            next_token_probs = torch.softmax(logits[0, -1, :], dim=0)

            # Get probability of the target token
            prob = next_token_probs[target_token_id].item()
            token_probs.append(prob)

            # Accumulate log probability (more numerically stable)
            log_prob_sum += torch.log(next_token_probs[target_token_id]).item()

            # Append this token to the sequence for next iteration
            if i < len(target_tokens) - 1:  # Don't extend on last iteration
                current_input_ids = torch.cat([
                    current_input_ids,
                    torch.tensor([[target_token_id]], device=device)
                ], dim=1)

                # Update inp dictionary
                inp = {
                    'input_ids': current_input_ids,
                    'attention_mask': torch.ones_like(current_input_ids)
                }

    # Convert log probability back to probability
    total_prob = torch.exp(torch.tensor(log_prob_sum)).item()

    return total_prob, token_probs





# Integration notes for the activation_patching.py script:
#
# 1. Modify calculate_hidden_flow to accept target_tokens (list) instead of target_token (int)
# 2. Modify trace_with_patch to handle multi-token targets autoregressively
# 3. Update the __main__ block to pass all target tokens instead of just the first one
#
# Key changes needed:
# - Instead of: target_token_id = target_tokens[0]
# - Use: target_token_ids = target_tokens  # Pass the full list
#
# Then modify trace_with_patch to loop through tokens autoregressively,
# computing P(token_i | token_1, ..., token_{i-1}) for each token.


# Example usage:
if __name__ == "__main__":
    from causal_trace import ModelAndTokenizer, make_inputs
    import numpy as np

    # Load model
    mt = ModelAndTokenizer("gpt2", torch_dtype=torch.float16)

    # Example: compute probability of multi-token word "basketball"
    prompt = "She spoke the language fluently and understood the local customs so she"
    target_word = " assimilated"

    # Tokenize
    target_tokens = mt.tokenizer.encode(target_word, add_special_tokens=False)

    print(f"Prompt: '{prompt}'")
    print(f"Target word: '{target_word}'")
    print(f"Target tokens: {target_tokens}")
    print(f"Target token strings: {[mt.tokenizer.decode([t]) for t in target_tokens]}")

    # Create input using make_inputs
    inp = make_inputs(mt.tokenizer, [prompt])

    # Calculate probability
    total_prob, token_probs = calculate_multitoken_probability(
        mt.model, mt.tokenizer, inp, target_tokens
    )

    print(f"\nIndividual token probabilities:")
    for i, (tok_id, prob) in enumerate(zip(target_tokens, token_probs)):
        tok_str = mt.tokenizer.decode([tok_id])
        print(f"  Token {i + 1}: '{tok_str}' -> P = {prob:.6f}")

    print(f"\nJoint probability P('{target_word}') = {total_prob:.6f}")
    print(f"Product of individual probs = {np.prod(token_probs):.6f}")

    # Show log probability (more interpretable for very small probabilities)
    log_prob = np.sum([np.log(p) for p in token_probs])
    print(f"Log probability = {log_prob:.4f}")

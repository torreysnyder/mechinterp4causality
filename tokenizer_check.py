from transformers import GPT2Tokenizer

# Load GPT2-small tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

print("Enter words to get their GPT-2 token IDs. Type 'done' to exit.\n")

while True:
    word = input("Enter a word: ").strip()
    if word.lower() == "done":
        print("Exiting...")
        break

    # Encode the word to get token IDs
    token_ids = tokenizer.encode(word, add_special_tokens=False)

    # Display results
    print(f"Input word: {word}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded tokens: {[tokenizer.decode([tid]) for tid in token_ids]}")
    print("-" * 40)


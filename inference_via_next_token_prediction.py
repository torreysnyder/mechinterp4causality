import torch
import hydra
from pathlib import Path
from tqdm import tqdm
from causal_transformers.utils import model_utils
from causal_transformers.dataset.preprocessor import Preprocessor

PROMPT_FILES = [
    #"causal_transformers/inference/intervention.txt",
    #"causal_transformers/inference/scm_index.txt",
    "causal_transformers/inference/variable_value.txt"
]

CHECKPOINT_PATH = (
"causal_transformers/checkpoints/gaussian/gaussian_dataset_v6/hooked_transformer_deep/instance3/checkpoint_0300.pth"
)

def load_model(checkpoint_path, cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_utils.get_hooked_transformer_model(cfg, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from (epoch {checkpoint['epoch']}, "
          f"train loss {checkpoint['stats']['train']['loss']:.6f})")
    return model, device

@torch.no_grad()
def predict_next_token(model, preprocessor, prompt: str, device) -> str:
    indices_tensor = preprocessor(prompt)
    tokens = preprocessor.tokenize(prompt)
    prompt_len = len(tokens)
    input_ids = torch.tensor([preprocessor.vocab[t] for t in tokens], device=device).unsqueeze(0)
    vocab_size = model.cfg.d_vocab
    if (input_ids < 0).any() or (input_ids >= vocab_size).any():
        raise ValueError(
            f"Token IDs out of range [0, {vocab_size}): "
            f"f[{input_ids.min()}, {input_ids.max()}]"
        )
    logits = model(input_ids)
    next_token_id = logits[0, -1, :].argmax().item()
    next_token_str = preprocessor.vocab_inv[next_token_id]
    return next_token_str

def process_file(path: Path, model, preprocessor, device):
    lines = [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) % 2 != 0:
        raise ValueError(f"{path}: expected an even number of lines, got {len(lines)}")
    new_lines = []
    skipped = 0
    for i in tqdm(range(0, len(lines), 2), desc=path.name):
        clean_line = lines[i]
        corrupt_line = lines[i + 1]
        try:
            predicted_token = predict_next_token(model, preprocessor, clean_line, device)
        except Exception as e:
            print(f" [warn] pair {i//2 + 1}: prediction failed ({e}), skipping")
            skipped += 1
            new_lines.append(clean_line)
            new_lines.append(corrupt_line)
            continue
        new_lines.append(clean_line + " " + predicted_token)
        new_lines.append(corrupt_line + " " + predicted_token)
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    n_pairs = len(lines) // 2
    print(f"wrote {n_pairs - skipped}/{n_pairs} pairs ({skipped}) [{path}]")
    print("Sample (first 2 pairs):")
    for ln in new_lines[:4]:
        print(f" {ln}")

@hydra.main(config_path="causal_transformers/config", config_name="config", version_base=None)
def main(cfg):
    model, device = load_model(CHECKPOINT_PATH, cfg)
    preprocessor = Preprocessor(cfg)
    print(f"\nProcessing {len(PROMPT_FILES)} prompt files...\n")
    for fname in PROMPT_FILES:
        path = Path(fname)
        if not path.exists():
            print(f"[skip] {fname} not found")
            continue
        print(f"\n{'='}*60")
        print(f"File: {fname}")
        process_file(path, model, preprocessor, device)
    print("\nAll done.")

if __name__ == "__main__":
    main()


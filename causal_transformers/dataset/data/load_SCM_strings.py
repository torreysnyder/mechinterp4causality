import torch
import json
import math
from pathlib import Path

def tensor_to_list(obj):
    """Recursively convert tensors/numpy to JSON-serialisable Python, NaN -> null."""
    if isinstance(obj, torch.Tensor):
        return tensor_to_list(obj.tolist())
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, list):
        return [tensor_to_list(v) for v in obj]
    return obj

# Resolve paths relative to this script's location, not the working directory.
# Script lives at: causal_transformers/dataset/data/load_SCM_strings.py
# So SCRIPT_DIR  = causal_transformers/dataset/data/
SCRIPT_DIR = Path(__file__).parent.resolve()

# Load the .pt file
pt_path = SCRIPT_DIR / "datasets/gaussian_dataset_v6/worlds/world_weights.pt"
data = torch.load(pt_path, weights_only=False, map_location="cpu")

# Convert everything to JSON-serialisable types
json_data = tensor_to_list(data)


# Save as JSON
# Output path: causal_transformers/inference/SCM_strings.json
json_path = SCRIPT_DIR.parent.parent / "inference/SCM_strings.json"
json_path.parent.mkdir(parents=True, exist_ok=True)
with open(json_path, "w") as f:
    json.dump(json_data, f, indent=2)

print(f"Loaded:  {pt_path}")
print(f"Saved:   {json_path}")
print("Done.")

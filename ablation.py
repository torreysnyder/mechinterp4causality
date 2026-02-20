import torch
from causal_transformers.utils import model_utils
import hydra
from omegaconf import OmegaConf


def load_finetuned_model(checkpoint_path, config_path=None, cfg=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg is None:
        assert config_path is not None, "Must provide either cfg or config_path"
        cfg = OmegaConf.load(config_path)
    model = model_utils.get_hooked_transformer_model(cfg, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model laoded from checkpoint at epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['stats']['train']['loss']:.6f}")
    return model, checkpoint

if __name__ == "__main__":
    @hydra.main(config_path="causal_transformers/config", config_name="config", version_base=None)
    def main(cfg):
        checkpoint_path = "causal_transformers/checkpoints/gaussian/gaussian_dataset_v6/hooked_transformer_deep/instance3/checkpoint_0300.pth"
        model, checkpoint = load_finetuned_model(checkpoint_path, cfg=cfg)
    main()

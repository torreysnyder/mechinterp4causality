import shutil
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import torch
from causal_transformers.utils.config_utils import get_config_entry


def checkpoint_elements_to_str(cfg: DictConfig):
    s = ""
    for key_string in cfg.checkpoint_elements:
        s += "-" + key_string + "=" + str(get_config_entry(cfg, key_string))
    return s


def get_checkpoint_dir(cfg: DictConfig, probe=False):
    model = cfg.model.name
    model += checkpoint_elements_to_str(cfg)
    final_dir = "instance" + str(cfg.train.instance)
    checkpoints_str = 'checkpoints_probe' if probe else 'checkpoints'
    checkpoint_dir = (
        Path("causal_transformers")
        / checkpoints_str
        / cfg.experiment_name
        / cfg.dataset.name
        / model
        / final_dir
    )
    if probe:
        probe_dir = f"{cfg.probe.name}"
        for token_type in cfg.probe.token_types:
            probe_dir += f"_{token_type}"
        checkpoint_dir = checkpoint_dir / probe_dir
    return checkpoint_dir


def get_checkpoint_path_from_dir(cfg, checkpoints_dir: Path):
    glob_str = 'checkpoint_*.pth'
    checkpoints = list(checkpoints_dir.glob(glob_str))
    if len(checkpoints) == 0:
        print("WARNING: No checkpoints found in directory: ", checkpoints_dir)
        return None
    if cfg.checkpoint_epoch != -1:
        checkpoint = checkpoints_dir / f"checkpoint_{cfg.checkpoint_epoch:04d}.pth"
        if checkpoint.exists():
            return checkpoint
        else:
            raise FileNotFoundError(f"Checkpoint {checkpoint} not found.")
    checkpoints.sort()
    checkpoint = checkpoints[-1]
    return checkpoint


def get_checkpoint_path(cfg: DictConfig, probe=False):
    checkpoint_dir = get_checkpoint_dir(cfg, probe=probe)
    path = get_checkpoint_path_from_dir(cfg, checkpoint_dir)
    return path


def prep_checkpoints(cfg, model, optimizer, lr_scheduler=None, probe=False):
    """
    Prepare checkpoints directory and optionally load the latest checkpoint.

    Behaviour:
    - If cfg.train.remove_checkpoints is True:
        * Delete any existing checkpoint directory and start from epoch 1.
    - Else:
        * If a checkpoint is found:
            - If last_epoch >= cfg.train.epochs:
                * Do NOT load the checkpoint (to avoid 'Epochs to train: 0').
                * Start a fresh run from epoch 1 using the model as currently initialised.
            - Else:
                * Load model / optimizer / scheduler state and resume from last_epoch + 1.
    """
    checkpoint_dir = get_checkpoint_dir(cfg, probe)
    start_epoch = 1

    if cfg.train.remove_checkpoints:
        if checkpoint_dir.exists():
            print("Removing existing checkpoint directory: ", checkpoint_dir)
            shutil.rmtree(checkpoint_dir)
    else:
        latest_checkpoint = get_checkpoint_path(cfg, probe=probe)
        if latest_checkpoint is not None:
            checkpoint = torch.load(latest_checkpoint, weights_only=False)
            last_epoch = checkpoint.get('epoch', 0)

            if last_epoch >= cfg.train.epochs:
                # We have already trained up to (or beyond) the configured total epochs.
                # If we were to resume, there would be 0 epochs left to train.
                print(
                    f"Found checkpoint at epoch {last_epoch}, which is >= "
                    f"configured total epochs ({cfg.train.epochs})."
                )
                print(
                    "Starting a NEW training run from epoch 1 without loading the "
                    "checkpoint. If you intended to continue training beyond this, "
                    "increase cfg.train.epochs."
                )
                # NOTE: We deliberately do NOT load model/optimizer/scheduler state here;
                # the model stays in its freshly-initialised state.
                start_epoch = 1
            else:
                # Normal resume: load states and continue from last_epoch + 1
                model.load_state_dict(checkpoint['model_state_dict'])
                if optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if lr_scheduler is not None and checkpoint.get('lr_scheduler_state_dict') is not None:
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                start_epoch = last_epoch + 1

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, start_epoch


def save_model(model, optimizer, lr_scheduler, epoch, stats, cfg, checkpoint_dir, checkpoint_name):
    config_dict = OmegaConf.to_container(
        cfg,
        resolve=True,
        enum_to_str=True,
        structured_config_mode="dict"
    )
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
            'stats': stats,
            'config': config_dict
        },
        checkpoint_dir / checkpoint_name
    )

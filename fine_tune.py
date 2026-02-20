import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import hydra
import causal_transformers as ct
from causal_transformers.utils import config_utils, misc_utils, dataset_utils, model_utils, checkpoint_utils
from causal_transformers.analysis import accuracy, gaussian_accuracy
from causal_transformers.dataset import preprocessor
from tqdm import tqdm
from collections import OrderedDict
from torch.nn.utils import parameters_to_vector
import numpy as np
import matplotlib.pyplot as plt


def train_epoch(cfg, model, dataloader, device, optimizer, epoch, split):
    """
    Train or evaluate the model for one epoch
    Args:
    :param cfg: Hydra configuration object
    :param model: neural network model (hooked transformer model)
    :param dataloader: DatLoader for the current split
    :param device: torch device (CUDA)
    :param optimizer: Optimizer for training
    :param epoch: Current epoch number
    :param split: Dataset split ("train", "valid", "test")
    :return:
    stats_epoch: Dictionary containing loss and accuracy metrics
    """
    loss_epoch = 0.0
    stats_epoch = {}
    # Calculate and store the L2 norm of model parameters (normalized by number of parameters)
    param_vector = parameters_to_vector(model.parameters())
    stats_epoch["weight_norm"] = (param_vector.norm(2) / len(param_vector)).item()
    # Set model to training or evaluation mode based on the split
    train = split == "train"
    if train:
        model.train()
    else:
        model.eval()
    # Create progress bar for visual feedback during training
    progress_bar = tqdm(dataloader, desc=f"epoch {epoch} ({split})", unit="batch")
    # Initialize the appropriate accuracy tracker based on dataset type
    if cfg.dataset.type == "additive":
        accuracy = ct.analysis.accuracy.Accuracy(cfg, device)
    elif cfg.dataset.type == "gaussian":
        accuracy = ct.analysis.gaussian_accuracy.GaussianAccuracy(cfg, device)
    else:
        raise ValueError(f"dataset type {cfg.dataset.type} not recognized")
    # Iterate through batches
    for batch_idx, data in enumerate(progress_bar):
        indices, meta_data = data
        indices = indices.to(device)
        # Zero gradients only during training
        if train:
            optimizer.zero_grad()
        # Prepare inputs and targets for next-token prediction
        # inputs: all tokens except the last one
        # targets: all tokens except the first one (shifted by 1)
        inputs = indices[:, :-1]
        targets = indices[:, 1:]
        # Forward pass: get model predictions
        logits = model(inputs)
        # Calculate cross-entropy loss, ignoring padding tokens
        # Permute logits from (batch, seq_len, vocab_size) to (batch, vocab_size, seq_len)
        loss = nn.CrossEntropyLoss(ignore_index=ct.dataset.preprocessor.PAD_TOKEN)(logits.permute(0, 2, 1), targets)
        # Compute accuracy if configured or if evaluating on valid/test sets
        if cfg.train.compute_accuracy or split in ["test"]:
            accuracy.update(logits, inputs, targets, meta_data)
        # Backward pass and optimization (only during training)
        if train:
            loss.backward()
            optimizer.step()
        # Accumulate loss for epoch averaging
        loss_epoch += loss.item()
        # Update progress bar with current metrics
        postfix = OrderedDict(
            loss=loss_epoch / (batch_idx + 1),  # Running average of loss
            **accuracy.get_accuracy(),  # Current accuracy metrics
            weight_norm=stats_epoch["weight_norm"]
        )
        progress_bar.set_postfix(postfix)
    # Calculate final epoch statistics
    stats_epoch["loss"] = loss_epoch / len(dataloader)
    stats_epoch.update(accuracy.get_accuracy())
    return stats_epoch


def plot_loss_curves(train_losses, valid_losses, checkpoint_dir):
    """
    Plot and save training and validation loss curves
    Args:
    :param train_losses: List of training losses per epoch
    :param valid_losses: List of validation losses per epoch
    :param checkpoint_dir: Directory to save the plot
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    #plt.plot(epochs, valid_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plot_path = checkpoint_dir / 'loss_curves_5.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Loss curve plot saved to: {plot_path}")
    plt.close()


@hydra.main(config_path="causal_transformers/config", config_name="config", version_base=None)
def train(cfg):
    """
    Main training function that orchestrates the entire training loop
    Uses Hydra for configuration management
    :param cfg: Hydra configuration object
    :return: None
    """
    # Print configuration for visibility
    ct.utils.config_utils.pretty_print_cfg(cfg)
    # Set random seeed for reproducibility
    ct.utils.misc_utils.set_random_seed(cfg.train.instance)
    # Set up device (GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    # Load training, validation and test dataloaders
    print('Loading data')
    dataloaders = ct.utils.dataset_utils.get_dataloaders(cfg, batch_size=cfg.train.batch_size)
    print('Data loaded')
    # Initialize model with hooks for analysis/interpretability
    model = ct.utils.model_utils.get_hooked_transformer_model(cfg, device)
    # Initialize AdamW optimizer with parameters from config
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.adam.lr, betas=cfg.train.adam.betas,
                            eps=cfg.train.adam.eps, weight_decay=cfg.train.adam.weight_decay)
    # Initialiize learning rate scheduler (reduces LR by gamma every step_size epochs)
    scheduler = StepLR(optimizer, step_size=cfg.train.step_lr.step_size, gamma=cfg.train.step_lr.gamma)
    # Prepare checkpoint directory and load from checkpoint if resuming training
    checkpoint_dir, start_epoch = ct.utils.checkpoint_utils.prep_checkpoints(cfg, model, optimizer, scheduler)
    # reset learning rate and scheduler step size (in case loaded from checkpoint)
    optimizer.param_groups[0]['lr'] = cfg.train.adam.lr
    scheduler.step_size = cfg.train.step_lr.step_size
    print(scheduler.step_size)
    # initialize TensorBoard writer for logging metrics
    writer = SummaryWriter(checkpoint_dir / f'tensorboard')

    # Initialize lists to track losses for plotting
    train_losses = []
    valid_losses = []

    # Debug: Print training configuration
    print(f"\n{'=' * 60}")
    print(f"TRAINING CONFIGURATION:")
    print(f"  Start epoch: {start_epoch}")
    print(f"  Total epochs: {cfg.train.epochs}")
    print(f"  Epochs to train: {cfg.train.epochs + 1 - start_epoch}")
    print(f"  Checkpoint interval: {cfg.train.checkpoint_interval}")
    print(f"  Batch size: {cfg.train.batch_size}")
    print(f"{'=' * 60}\n")

    # Main training loop
    eval_interval = getattr(cfg.train, "eval_interval", 5)  # default: every 5 epochs

    # Main training loop
    for epoch in range(start_epoch, cfg.train.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch}/{cfg.train.epochs}")
        print(f"{'=' * 60}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        stats = {}

        # -------------------------
        # 1) TRAINING PHASE (epoch)
        # -------------------------
        print("\nStarting train phase...")
        stats["train"] = train_epoch(cfg, model, dataloaders["train"], device, optimizer, epoch, "train")
        print(f"Completed train phase - Loss: {stats['train']['loss']:.6f}")

        # log training metrics
        for key in stats["train"]:
            writer.add_scalar(f"train/{key}", stats["train"][key], epoch)

        # -------------------------
        # 2) (OPTIONAL) VALIDATION
        # -------------------------
        run_valid = (epoch == start_epoch) or (epoch % eval_interval == 0)

        if run_valid:
            print("\nStarting valid phase...")
            stats["valid"] = train_epoch(cfg, model, dataloaders["valid"], device, optimizer, epoch, "valid")
            print(f"Completed valid phase - Loss: {stats['valid']['loss']:.6f}")

            # log validation metrics
            for key in stats["valid"]:
                writer.add_scalar(f"valid/{key}", stats["valid"][key], epoch)

            # store losses
            train_losses.append(stats["train"]["loss"])
            valid_losses.append(stats["valid"]["loss"])
        else:
            # No validation this epoch
            train_losses.append(stats["train"]["loss"])
            valid_losses.append(float("nan"))  # keeps list aligned for plotting

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {stats['train']['loss']:.6f}")
        if "valid" in stats:
            print(f"  Valid Loss: {stats['valid']['loss']:.6f}")
        else:
            print("  Valid Loss: (not evaluated this epoch)")


        # save checkpoint at first epoch and at regular intervals
        if epoch == 1 or epoch % cfg.train.checkpoint_interval == 0:
            print(f"  Saving checkpoint...")
            ct.utils.checkpoint_utils.save_model(model, optimizer, scheduler, epoch, stats, cfg, checkpoint_dir,
                                                 "checkpoint_{:04d}.pth".format(epoch))
            print(f"  Checkpoint saved!")

        # Generate and save loss curve plot after each epoch
        plot_loss_curves(train_losses, valid_losses, checkpoint_dir)

        # update learning rate according to schedule
        scheduler.step()
        print(f"  Learning rate after step: {optimizer.param_groups[0]['lr']}")

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'=' * 60}\n")

    # Generate final loss curve plot
    plot_loss_curves(train_losses, valid_losses, checkpoint_dir)

    # clean up
    writer.close()
    print('Finished Training')


if __name__ == "__main__":
    train()

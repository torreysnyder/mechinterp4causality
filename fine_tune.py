import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import hydra
import causal_transformers as ct
from tqdm import tqdm
from collections import OrderedDict
from torch.nn.utils import parameters_to_vector
import numpy as np

def train_epoch(cfg, model, dataloader, device, optimizer, epoch, split):
    loss_epoch = 0.0
    stats_epoch = {}
    param_vector = parameters_to_vector(model.parameters())
    stats_epoch["weight_norm"] = (param_vector.norm(2)/len(param_vector)).item()
    train = split == "train"
    if train:
        model.train()
    else:
        model.eval()
    progress_bar = tqdm(dataloader, desc=f"epoch {epoch} ({split})", unit="batch")
    if cfg.dataset.type == "additive":
        accuracy = ct.analysis.Accuracy(cfg, device)
    elif cfg.dataset.type == "gaussian":
        accuracy = ct.analysis.GaussianAccuracy(cfg, device)
    else:
        raise ValueError(f"dataset type {cfg.dataset.type} not recognized")

    for batch_idx, data in enumerate(progress_bar):
        indices, meta_data = data
        indices = indices.to(device)
        if train:
            optimizer.zero_grad()
        inputs = indices[:, :-1]
        targets = indices[:, 1:]
        logits = model(inputs)
        loss = nn.CrossEntropyLoss(ignore_index=ct.dataset.preprocessor.PAD_TOKEN)(logits.permute(0, 2, 1), targets)
        if cfg.train.compute_accuracy or split in ["valid", "test"]:
            accuracy.update(logits, inputs, targets, meta_data)
        if train:
            loss.backward()
            optimizer.step()
        loss_epoch += loss.item()
        postfix = OrderedDict(
            loss=loss_epoch/(batch_idx + 1),
            **accuracy.get_accuracy(),
            weight_norm=stats_epoch["weight_norm"]
        )
        progress_bar.set_postfix(postfix)

    stats_epoch["loss"] = loss_epoch/len(dataloader)
    stats_epoch.update(accuracy.get_accuracy())
    return stats_epoch

@hydra.main(version_base=None, config_name='config', config_path=ct.paths.CONFIG_PATH)
def train(cfg):
    ct.utils.pretty_print_cfg(cfg)
    ct.utils.set_random_seed(cfg.train.instance)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    print('Loading data')
    dataloaders = ct.utils.get_dataloaders(cfg, batch_size=cfg.train.batch_size)
    print('Data loaded')
    model = ct.utils.get_hooked_transformer_model(cfg, device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.adam.lr, betas=cfg.train.adam.betas, eps=cfg.train.adam.eps, weight_decay=cfg.train.adam.weight_decay)
    scheduler = StepLR(optimizer, step_size=cfg.train.step_lr.step_size, gamma=cfg.train.step_lr.gamma)
    checkpoint_dir, start_epoch = ct.utils.prep_checkpoints(cfg, model, optimizer, scheduler)
    optimizer.param_groups[0]['lr'] = cfg.train.adam.lr
    scheduler.step_size = cfg.train.step_lr.step_size
    print(scheduler.step_size)
    writer = SummaryWriter(checkpoint_dir/f'tensorboard')
    for epoch in range(start_epoch, cfg.train.epochs+1):
        print(f"Epoch {epoch}/{cfg.train.epochs}")
        print("Learning rate: ", optimizer.param_groups[0]['lr'])
        stats = {}
        for split in ["train", "valid"]:
            stats[split] = train_epoch(cfg, model, dataloaders[split], device, optimizer, epoch, split)
            for key in stats[split]:
                writer.add_scalar(f'{split}/{key}', stats[split][key], epoch)
        if epoch == 1 or epoch % cfg.train.checkpoint_interval == 0:
            ct.utils.save_model(model, optimizer, scheduler, epoch, stats, cfg, checkpoint_dir, "checkpoint_{:04d}.pth".format(epoch))
        scheduler.step()
    writer.close()
    print('Finished Training')

if __name__ == "__main__":
    train()



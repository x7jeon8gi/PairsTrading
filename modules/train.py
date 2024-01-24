import math
import sys

import torch
import torch.nn.functional as F
import numpy as np


def train_epoch(
    model,
    criterion_ins,
    criterion_clu,
    data_loader,
    optimizer,
    scheduler,
    device,
    epoch,
    training_step,
    progress_bar,
    accelerator,
    use_accelerator,
    use_wandb
):
    model.train()
    loss_epoch = 0
    
    for batch_idx, (x_w, x_s) in enumerate(data_loader):
        training_step += 1
        optimizer.zero_grad()
        x_w = x_w.to(device)
        x_s = x_s.to(device)
        z_i, z_j, c_i, c_j = model(x_w, x_s)
        loss_instance = criterion_ins(torch.concat((z_i, z_j),dim=0))
        loss_cluster = criterion_clu(torch.concat((c_i, c_j), dim=0))
        loss = loss_instance + loss_cluster
        if use_accelerator:
            accelerator.backward(loss)
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        loss_epoch += loss.item()

        current_lr = scheduler.get_last_lr()[0]  # Assuming one param_group. If this doesn't work, try optimizer.param_groups[0]['lr']
        if use_wandb:
            accelerator.log({"learning_rate": current_lr}, step=training_step)

        progress_bar.update(1)
        if use_wandb:
            accelerator.log({"step_loss":loss, "step_cluster_loss":loss_cluster, "step_instance_loss": loss_instance}, step=training_step)

    progress_bar.set_description(f"Epoch {epoch} - Train Loss: {loss_epoch / len(data_loader):.4f}")
        
    return model, loss_epoch / len(data_loader), training_step


# 하면 할 수록 낮아지니까 딱히 evaluation을 처음에 두지 않은 듯
def valid_epoch(
    model,
    criterion_ins,
    criterion_clu,
    data_loader,
    device,
):

    model.eval()
    loss_epoch = 0
    
    with torch.no_grad():
        for batch_idx, (x_w, x_s) in enumerate(data_loader):
            x_w = x_w.to(device)
            x_s = x_s.to(device)
            z_i, z_j, c_i, c_j = model(x_w, x_s)
            loss_instance = criterion_ins(torch.concat((z_i, z_j),dim=0))
            loss_cluster = criterion_clu(torch.concat((c_i, c_j), dim=0))
            loss = loss_instance + loss_cluster
            loss_epoch += loss.item()
            
    return loss_epoch / len(data_loader)
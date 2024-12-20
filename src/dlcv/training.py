import gc
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from sklearn.metrics import jaccard_score
import numpy as np
from dlcv.utils import *
# from torch import GradScaler, autocast
import torch.amp

def train_one_epoch(model, data_loader, criterion, optimizer, device, scaler):
    model.train()
    epoch_loss = 0.0
    # optimizer.zero_grad()

    for inputs, _, class_masks in tqdm(data_loader, desc="Training"):
        inputs, class_masks = inputs.to(device), class_masks.to(device)
        optimizer.zero_grad()
        

        with torch.autocast(device.type):
            # Forward pass
            outputs_material = model(inputs)
            outputs_material = torch.clamp(outputs_material, min=-20, max=20)
            
            class_masks = class_masks.long()        
            # Calculate loss
            loss_material = criterion(outputs_material, class_masks)
        
        # Backward pass
        scaler.scale(loss_material).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss_material.item() * inputs.size(0)

    epoch_loss /= len(data_loader.dataset)
    return epoch_loss


def evaluate_one_epoch(model, data_loader, device, criterion, memory_cleanup_frequency=15):
    model.eval()
    epoch_loss = 0.0
    all_preds_material = []
    all_labels_material = []

    with torch.no_grad():
        for batch_idx, (inputs, _, class_masks) in enumerate(tqdm(data_loader, desc="Evaluation")):
            inputs, class_masks = inputs.to(device), class_masks.to(device)

            if class_masks.ndimension() == 4 and class_masks.size(1) == 1:
                class_masks = class_masks.squeeze(1)
                
            class_masks = class_masks.long()
            
            outputs_material = model(inputs)
            
            # Calculate loss
            loss_material = criterion(outputs_material, class_masks)
            
            # Get predictions
            _, predicted_material = torch.max(outputs_material, 1)
            
            # Collect predictions and labels
            all_preds_material.extend(predicted_material.cpu().numpy().flatten())
            all_labels_material.extend(class_masks.cpu().numpy().flatten())
            
            epoch_loss += loss_material.item() * inputs.size(0)

            # Periodic memory cleanup
            if (batch_idx + 1) % memory_cleanup_frequency == 0:
                del inputs, class_masks, outputs_material, predicted_material
                gc.collect()
                torch.cuda.empty_cache()

    # Final memory cleanup
    del inputs, class_masks, outputs_material, predicted_material
    gc.collect()
    torch.cuda.empty_cache()

    epoch_loss /= len(data_loader.dataset)
    
    all_preds_material = np.array(all_preds_material)
    all_labels_material = np.array(all_labels_material)

    num_classes = model.classifier_material.classifier[-1].out_channels
    iou_material = calculate_mean_iou(all_preds_material, all_labels_material, num_classes=num_classes)

    return epoch_loss, iou_material


def calculate_mean_iou(preds, labels, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        if union == 0:
            ious.append(float('nan'))  # If there is no union, the IoU is undefined
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)


def train_and_evaluate_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, scheduler=None, early_stopping=False):
    train_losses = []
    test_losses = []
    test_ious_material = []
    best_test_loss = float('inf')
    consecutive_no_improvement = 0
    scaler = torch.GradScaler('cuda')

    # Use low-level and high-level layers for unfreezing
    low_level_layers = model.backbone.low_level_layers
    high_level_layers = model.backbone.high_level_layers

    # Group the layers into blocks for gradual unfreezing
    mobilenet_blocks = [low_level_layers] + list(high_level_layers.children())
    total_blocks = len(mobilenet_blocks)

    # Start unfreezing after first half of epochs
    unfreeze_start_epoch = num_epochs // 2

    # Unfreeze one block every few epochs (integer-based)
    unfreeze_frequency = (num_epochs - unfreeze_start_epoch) // total_blocks
    if unfreeze_frequency == 0:
        unfreeze_frequency = 1  # Ensure that we at least unfreeze one block at a time if epochs are short
    
    for epoch in range(num_epochs):
        # Gradual unfreezing: progressively unfreeze blocks after unfreeze_start_epoch
        if epoch >= unfreeze_start_epoch:
            # Calculate how many blocks to unfreeze based on current epoch
            num_blocks_to_unfreeze = (epoch - unfreeze_start_epoch) // unfreeze_frequency + 1
            num_blocks_to_unfreeze = min(num_blocks_to_unfreeze, total_blocks)  # Avoid unfreezing more than available blocks
            
            for i in range(num_blocks_to_unfreeze):
                for param in mobilenet_blocks[i].parameters():
                    param.requires_grad = True
        else:
            # Freeze all blocks in the first unfreeze_start_epoch epochs
            for block in mobilenet_blocks:
                for param in block.parameters():
                    param.requires_grad = False

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        train_losses.append(train_loss)

        test_loss, test_iou_material = evaluate_one_epoch(model, test_loader, device, criterion)
        test_losses.append(test_loss)
        test_ious_material.append(test_iou_material)

        if scheduler is not None:
            scheduler.step()

        if early_stopping:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1

            if consecutive_no_improvement >= 4:
                print("Early stopping triggered!")
                break

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test IoU (Material): {test_iou_material:.4f}")

    return train_losses, test_losses, test_ious_material
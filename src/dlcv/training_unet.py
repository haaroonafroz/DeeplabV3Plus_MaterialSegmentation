import gc
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
from dlcv.utils import *
import torch.amp

def train_one_epoch_unet(model, data_loader, criterion, optimizer, device, scaler):
    model.train()
    epoch_loss = 0.0

    for inputs, _, class_masks in tqdm(data_loader, desc="Training U-Net"):
        inputs, class_masks = inputs.to(device), class_masks.to(device)
        # Resize class masks to match output shape
        class_masks = F.interpolate(class_masks.unsqueeze(1).float(), size=(128, 128), mode='bilinear', align_corners=False).squeeze(1).long()

        optimizer.zero_grad()

        with torch.autocast(device.type):
            # Forward pass
            outputs = model(inputs)
            # print("Outputs shape:", outputs.shape)
            outputs = torch.clamp(outputs, min=-20, max=20)

            class_masks = class_masks.long()
            # print("Class masks shape:", class_masks.shape)

            if class_masks.dim() == 3:  # If shape is [batch_size, height, width]
                class_masks = class_masks.unsqueeze(1)
            # Calculate loss
            loss = criterion(outputs, class_masks)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item() * inputs.size(0)

    epoch_loss /= len(data_loader.dataset)
    return epoch_loss


def evaluate_one_epoch_unet(model, data_loader, device, criterion, memory_cleanup_frequency=15):
    model.eval()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, _, class_masks) in enumerate(tqdm(data_loader, desc="Evaluation U-Net")):
            inputs, class_masks = inputs.to(device), class_masks.to(device)

            if class_masks.ndimension() == 4 and class_masks.size(1) == 1:
                class_masks = class_masks.squeeze(1)

            class_masks = class_masks.long()

            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, class_masks)

            # Get predictions
            _, predicted = torch.max(outputs, 1)

            # Collect predictions and labels
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_labels.extend(class_masks.cpu().numpy().flatten())

            epoch_loss += loss.item() * inputs.size(0)

            # Periodic memory cleanup
            if (batch_idx + 1) % memory_cleanup_frequency == 0:
                del inputs, class_masks, outputs, predicted
                gc.collect()
                torch.cuda.empty_cache()

    # Final memory cleanup
    del inputs, class_masks, outputs, predicted
    gc.collect()
    torch.cuda.empty_cache()

    epoch_loss /= len(data_loader.dataset)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    num_classes = model.final_conv.out_channels
    iou = calculate_mean_iou(all_preds, all_labels, num_classes=num_classes)

    return epoch_loss, iou


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


def train_and_evaluate_model_unet(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, scheduler=None, early_stopping=False):
    train_losses = []
    test_losses = []
    test_ious = []
    best_test_loss = float('inf')
    consecutive_no_improvement = 0
    scaler = torch.GradScaler('cuda')

    # Low-level and high-level layers of MobileNet encoder
    low_level_layers = model.encoder[0]  # Initial layers
    high_level_layers = model.encoder[1]  # High-level layers

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
            num_blocks_to_unfreeze = (epoch - unfreeze_start_epoch) // unfreeze_frequency + 1
            num_blocks_to_unfreeze = min(num_blocks_to_unfreeze, total_blocks)

            for i in range(num_blocks_to_unfreeze):
                for param in mobilenet_blocks[i].parameters():
                    param.requires_grad = True
        else:
            for block in mobilenet_blocks:
                for param in block.parameters():
                    param.requires_grad = False

        train_loss = train_one_epoch_unet(model, train_loader, criterion, optimizer, device, scaler)
        train_losses.append(train_loss)

        test_loss, test_iou = evaluate_one_epoch_unet(model, test_loader, device, criterion)
        test_losses.append(test_loss)
        test_ious.append(test_iou)

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

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}")

    return train_losses, test_losses, test_ious

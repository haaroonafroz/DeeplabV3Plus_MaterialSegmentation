import gc
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from sklearn.metrics import jaccard_score
import numpy as np
from dlcv.utils import cross_entropy_4d

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0

    for inputs, class_targets in tqdm(data_loader, desc="Training"):
        inputs, class_targets = inputs.to(device), class_targets.to(device)
        optimizer.zero_grad()

        outputs_material = model(inputs)
        
        loss_material = cross_entropy_4d(outputs_material, class_targets)
        
        loss_material.backward()
        optimizer.step()
        
        epoch_loss += loss_material.item() * inputs.size(0)

    epoch_loss /= len(data_loader.dataset)
    return epoch_loss


def evaluate_one_epoch(model, data_loader, device, memory_cleanup_frequency=20):
    model.eval()
    epoch_loss = 0.0
    all_preds_material = []
    all_labels_material = []

    with torch.no_grad():
        for batch_idx, (inputs, class_targets) in enumerate(tqdm(data_loader, desc="Evaluation")):
            inputs, class_targets = inputs.to(device), class_targets.to(device)

            if class_targets.ndimension() == 4 and class_targets.size(1) == 1:
                class_targets = class_targets.squeeze(1)
                
            class_targets = class_targets.long()
            
            outputs_material = model(inputs)
            
            loss_material = cross_entropy_4d(outputs_material, class_targets)
            
            _, predicted_material = torch.max(outputs_material, 1)
            
            all_preds_material.extend(predicted_material.cpu().numpy().flatten())
            all_labels_material.extend(class_targets.cpu().numpy().flatten())
            
            epoch_loss += loss_material.item() * inputs.size(0)

            # Periodic memory cleanup
            if (batch_idx + 1) % memory_cleanup_frequency == 0:
                del inputs, class_targets, outputs_material, predicted_material
                gc.collect()
                torch.cuda.empty_cache()

    # Final memory cleanup
    del inputs, class_targets, outputs_material, predicted_material
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

    for epoch in range(num_epochs):
        # Gradual unfreezing
        if epoch < num_epochs // 3:
            for param in model.backbone.parameters():
                param.requires_grad = False
        elif epoch < 2 * num_epochs // 3:
            for param in model.backbone.parameters():
                param.requires_grad = True

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        test_loss, test_iou_material = evaluate_one_epoch(model, test_loader, device)
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
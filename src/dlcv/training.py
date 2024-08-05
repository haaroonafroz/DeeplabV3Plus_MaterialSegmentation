import gc
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from sklearn.metrics import jaccard_score
import numpy as np
from dlcv.utils import cross_entropy_4d
import torch_xla.core.xla_model as xm
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import jaccard_score
import numpy as np
from dlcv.utils import cross_entropy_4d

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0

    for inputs, targets, class_targets in tqdm(data_loader, desc="Training"):
        inputs, targets, class_targets = inputs.to(device), targets.to(device), class_targets.to(device)
        optimizer.zero_grad()

        outputs_object, outputs_material = model(inputs)
        
        loss_object = cross_entropy_4d(outputs_object, targets)
        loss_material = cross_entropy_4d(outputs_material, class_targets)
        
        loss = loss_object + loss_material
        
        loss.backward()
        if 'COLAB_TPU_ADDR' in os.environ:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        
        epoch_loss += loss.item() * inputs.size(0)

    epoch_loss /= len(data_loader.dataset)
    return epoch_loss


def evaluate_one_epoch(model, data_loader, device, memory_cleanup_frequency=20):
    model.eval()
    epoch_loss = 0.0
    all_preds_object = []
    all_labels_object = []
    all_preds_material = []
    all_labels_material = []

    with torch.no_grad():
        for batch_idx, (inputs, targets, class_targets) in enumerate(tqdm(data_loader, desc="Evaluation")):
            inputs, targets, class_targets = inputs.to(device), targets.to(device), class_targets.to(device)

            if targets.ndimension() == 4 and targets.size(1) == 1:
                targets = targets.squeeze(1)
            if class_targets.ndimension() == 4 and class_targets.size(1) == 1:
                class_targets = class_targets.squeeze(1)
                
            targets = targets.long()
            class_targets = class_targets.long()
            
            outputs_object, outputs_material = model(inputs)
            
            loss_object = cross_entropy_4d(outputs_object, targets)
            loss_material = cross_entropy_4d(outputs_material, class_targets)
            
            loss = loss_object + loss_material

            _, predicted_object = torch.max(outputs_object, 1)
            _, predicted_material = torch.max(outputs_material, 1)
            
            all_preds_object.extend(predicted_object.cpu().numpy().flatten())
            all_labels_object.extend(targets.cpu().numpy().flatten())
            all_preds_material.extend(predicted_material.cpu().numpy().flatten())
            all_labels_material.extend(class_targets.cpu().numpy().flatten())
            
            epoch_loss += loss.item() * inputs.size(0)

             # Periodic memory cleanup
            if (batch_idx + 1) % memory_cleanup_frequency == 0:
                del inputs, targets, class_targets, outputs_object, outputs_material, predicted_object, predicted_material
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif 'COLAB_TPU_ADDR' in os.environ:
                    xm.collect_xla_gc()


    # Final memory cleanup
    del inputs, targets, class_targets, outputs_object, outputs_material, predicted_object, predicted_material
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif 'COLAB_TPU_ADDR' in os.environ:
        xm.collect_xla_gc()

    epoch_loss /= len(data_loader.dataset)
    
    all_preds_object = np.array(all_preds_object)
    all_labels_object = np.array(all_labels_object)
    all_preds_material = np.array(all_preds_material)
    all_labels_material = np.array(all_labels_material)

    iou_object = calculate_mean_iou(all_preds_object, all_labels_object, num_classes=model.classifier_object.classifier[4].out_channels)
    iou_material = calculate_mean_iou(all_preds_material, all_labels_material, num_classes=model.classifier_material.classifier[4].out_channels)

    return epoch_loss, iou_object, iou_material


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
    test_ious_object = []
    test_ious_material = []
    best_test_loss = float('inf')
    consecutive_no_improvement = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        test_loss, test_iou_object, test_iou_material = evaluate_one_epoch(model, test_loader, device)
        test_losses.append(test_loss)
        test_ious_object.append(test_iou_object)
        test_ious_material.append(test_iou_material)

        if scheduler is not None:
            scheduler.step()

        if early_stopping:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1

            if consecutive_no_improvement >= 2:
                print("Early stopping triggered!")
                break

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test IoU (Object): {test_iou_object:.4f}, Test IoU (Material): {test_iou_material:.4f}")

    return train_losses, test_losses, test_ious_object, test_ious_material


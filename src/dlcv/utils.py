import csv
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns
from dlcv.config import *
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops

## CREATE CONFIG FUNCTION

def cfg_node_to_dict(cfg_node):
    """Convert a yacs CfgNode to a dictionary."""
    if not isinstance(cfg_node, CN):
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = cfg_node_to_dict(v)
        return cfg_dict

def create_config(run_name, backbone, base_lr, batch_size, num_epochs,
                   horizontal_flip_prob, rotation_degrees, crop_size, milestones, gamma,
                    early_stopping, pretrained_weights= '', save_path='',
                    root='/kaggle/input/material-dataset-2/Material_dataset',
                    config_dir='/kaggle/working/create_config'):
    # Get default configuration
    cfg = get_cfg_defaults()

    # Update the configuration with provided arguments
    cfg.DATA.MATERIAL_ROOT = root
    cfg.MISC.RUN_NAME = run_name
    cfg.MODEL.BACKBONE = backbone
    cfg.TRAIN.BASE_LR = base_lr
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TRAIN.NUM_EPOCHS = num_epochs
    cfg.AUGMENTATION.HORIZONTAL_FLIP_PROB = horizontal_flip_prob
    cfg.AUGMENTATION.ROTATION_DEGREES = rotation_degrees
    cfg.AUGMENTATION.CROP_SIZE = crop_size
    cfg.TRAIN.MILESTONES = milestones
    cfg.TRAIN.GAMMA = gamma
    cfg.MISC.PRETRAINED_WEIGHTS = pretrained_weights
    cfg.MISC.SAVE_PREDICTION = save_path
    cfg.TRAIN.EARLY_STOPPING = early_stopping

    # Ensure the config directory exists
    os.makedirs(config_dir, exist_ok=True)

    # Define the path for the new config file
    config_file_path = os.path.join(config_dir, f'{run_name}.yaml')
    #os.makedirs(config_file_path, exist_ok=True)

     # Convert the config object to a dictionary
    cfg_dict = cfg_node_to_dict(cfg)
    
    # Save the updated configuration to a YAML file
    with open(config_file_path, 'w') as config_file:
        yaml.dump(cfg_dict, config_file, default_flow_style=False)

    print(f"Config file saved at: {config_file_path}")
    with open(config_file_path, 'r') as file:
        print(file.read())
    return config_file_path

##---------------------------------------------------------------------------

# LOSS FUNCTION

def dice_loss(preds, targets, smooth=1.0):
    """
    Compute the Dice Loss between predictions and targets.
    
    Args:
        preds (Tensor): Predictions from the model of shape [batch_size, num_classes, height, width].
        targets (Tensor): Ground truth labels of shape [batch_size, height, width].
        smooth (float): Smoothing factor to avoid division by zero.
        
    Returns:
        Tensor: Computed Dice Loss.
    """
     # One-hot encode the targets
    num_classes = preds.size(1)

    # Ensure targets are one-hot encoded and match the number of classes
    if targets.ndimension() == 3:
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()  # Shape: [batch_size, num_classes, height, width]
    elif targets.ndimension() == 4 and targets.size(1) == 1:
        # If the target has a single channel dimension, squeeze it and one-hot encode
        targets = targets.squeeze(1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    else:
        raise ValueError("Unexpected target dimensions: expected 3D or 4D tensor.")
    
    # Apply softmax to predictions to get class probabilities
    # preds = F.softmax(preds, dim=1)
    
    # Calculate Dice coefficient
    intersection = (preds * targets_one_hot).sum(dim=[0, 2, 3])
    union = preds.sum(dim=[0, 2, 3]) + targets_one_hot.sum(dim=[0, 2, 3]) + smooth
    
    dice = (2. * intersection + smooth) / (union)
    dice = dice.mean()  # Average over all classes
    
    return 1 - dice

def cross_entropy_4d(input, target, smoothing = 0.1):
    """
    Custom cross-entropy loss function for segmentation that supports 4D targets.
    
    Args:
        input (Tensor): Predictions from the model of shape [batch_size, num_classes, height, width].
        target (Tensor): Ground truth labels of shape [batch_size, height, width].
    
    Returns:
        Tensor: Computed loss.
    """

    # If target has a channel dimension, squeeze it
    if target.ndimension() == 4 and target.size(1) == 1:
        target = target.squeeze(1)
    
    # Ensure the target is of type Long
    target = target.long() 

    # Flatten the tensors to be 2D for compatibility with F.cross_entropy
    # input needs to be [N, C] where N is the number of pixels and C is the number of classes
    # target needs to be [N] where N is the number of pixels
    input = input.permute(0, 2, 3, 1).contiguous().view(-1, input.size(1))
    target = target.view(-1)

     # Check for any invalid values in input or target
    if torch.isnan(input).any() or torch.isinf(input).any():
        print("Invalid input detected in cross_entropy")
    if torch.isnan(target).any() or torch.isinf(target).any():
        print("Invalid target detected in cross_entropy")

    # Number of classes
    num_classes = input.size(1)

    # Create one-hot encoding of targets
    one_hot = torch.zeros_like(input).scatter_(1, target.unsqueeze(1), 1)

    # Apply label smoothing
    one_hot = one_hot * (1 - smoothing) + (smoothing / num_classes)

    # Log probabilities
    log_probs = F.log_softmax(input, dim=1)

    # Calculate the loss using the smoothed labels
    loss = - (one_hot * log_probs).sum(dim=1)

    # Handle ignore_index if necessary
    if target.eq(0).any():
        # Set losses corresponding to ignore_index to zero
        loss[target.eq(0)] = 0

    return loss.mean()

def combined_loss(preds, targets, alpha=0.5):
    """
    Compute the combined Cross-Entropy and Dice Loss.
    
    Args:
        preds (Tensor): Predictions from the model of shape [batch_size, num_classes, height, width].
        targets (Tensor): Ground truth labels of shape [batch_size, height, width].
        alpha (float): Weight for Cross-Entropy Loss.
        
    Returns:
        Tensor: Computed combined loss.
    """
    # Compute Cross-Entropy Loss
    ce_loss = cross_entropy_4d(preds, targets)
    
    # Compute Dice Loss
    dice = dice_loss(preds, targets)

    # Check for NaNs
    if torch.isnan(ce_loss) or torch.isnan(dice):
        print(f"NaN detected in loss calculations: CE Loss: {ce_loss}, Dice Loss: {dice}")
        return torch.tensor(float('nan'), device=preds.device)  # Return nan to propagate through the training process
    
    # Combine the losses
    loss = alpha * ce_loss + (1 - alpha) * dice
    
    return loss

##-----------------------------------------------------------------------------------

def load_pretrained_weights(network, weights_path, device):
    """
    Loads pretrained weights (state_dict) into the specified network.

    Args:
        network (nn.Module): The network into which the weights are to be loaded.
        weights_path (str or pathlib.Path): The path to the file containing the pretrained weights.
        device (torch.device): The device on which the network is running (e.g., 'cpu' or 'cuda').
    Returns:
        network (nn.Module): The network with the pretrained weights loaded and adjusted if necessary.
    """
        # Load the checkpoint from the saved file
    #print(network.state_dict().keys())
    checkpoint = torch.load(weights_path, map_location=device)
    # print(f"Keys in checkpoint: {checkpoint.keys()}")
    
    # Extract the model state_dict from the checkpoint
    loaded_keys = []
    if 'model_state' in checkpoint:
        model_state = checkpoint['model_state']
        for key in model_state:
            if key in network.state_dict():
                network.state_dict()[key].copy_(model_state[key])
                loaded_keys.append(key)
            # else:
            #     print(f"Warning: Key '{key}' in model_state not found in network's state_dict.")

    if loaded_keys:
        print(f"Loaded keys: {loaded_keys}")
    
    return network


def freeze_layers(network, frozen_layers):
    """
    Freezes the specified layers of a network. Freezing a layer means its parameters will not be updated during training.

    Args:
        network (nn.Module): The neural network to modify.
        frozen_layers (list of str): A list of layer identifiers whose parameters should be frozen.
    """
    for name, param in network.named_parameters():
        if any(layer_name in name for layer_name in frozen_layers):
            param.requires_grad = False



def save_model(model, path):
    """
    Saves the model state_dict to a specified file.

    Args:
        model (nn.Module): The PyTorch model to save. Only the state_dict should be saved.
        path (str): The path where to save the model. Without the postifix .pth
    """
    os.makedirs(path, exist_ok=True)
    # Ensure device independence
    model_state = model.to(torch.device('cpu')).state_dict()
    torch.save(model_state, path + ".pth")


def get_stratified_param_groups(network, base_lr=0.001, stratification_rates=None):
    """
    Creates parameter groups with different learning rates for different layers of the network.

    Args:
        network (nn.Module): The neural network for which the parameter groups are created.
        base_lr (float): The base learning rate for layers not specified in stratification_rates.
        stratification_rates (dict): A dictionary mapping layer names to specific learning rates.

    Returns:
        param_groups (list of dict): A list of parameter group dictionaries suitable for an optimizer.
                                     Outside of the function this param_groups variable can be used like:
                                     optimizer = torch.optim.Adam(param_groups)
    """
    param_groups = []
    for name, param in network.named_parameters():
        for layer_name, lr in stratification_rates.items():
            if name.startswith(layer_name):
                param_groups.append({'params': param, 'lr': lr})
                break
        else:
            param_groups.append({'params': param, 'lr': base_lr})
    return param_groups


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def decode_segmap_1(image, nc=21):
    label_colors = voc_cmap(256)

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

def decode_segmap(image, num_classes=5):  # Adjust num_classes if you have 4 materials + background
    label_colors = np.array([
        [0, 0, 0],        # Black background
        [0, 0, 255],      # Blue - Metal
        [0, 255, 0],      # Green - Glass
        [255, 255, 0],    # Yellow - Plastic
        [255, 0, 0]       # Red - Wood
    ])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    # Print unique class indices
    unique_classes = np.unique(image)
    print(f"Unique predicted classes: {unique_classes}")

    for l in range(num_classes):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

def display_confidence_score(ax, output_softmax, predicted_mask, material_name, class_index):
    """
    Display the confidence score on the image for a specific class, without altering the heatmap visualization.

    Args:
        ax (matplotlib.axes): The axis to display the confidence map.
        output_softmax (np.array): The softmax output (shape [num_classes, H, W]).
        predicted_mask (np.array): Boolean mask where predicted class equals the current class.
        material_name (str): The name of the material.
        class_index (int): The index of the class to display confidence for.
    """
    # Extract the confidence map for the current class
    confidence_map = output_softmax[class_index, :, :]  # Correct indexing for 3D array

    # Visualize the confidence map with the heatmap
    heatmap = ax.imshow(confidence_map, cmap='hot', interpolation='nearest')
    ax.set_title(f"{material_name} Confidence")
    ax.axis("off")
    # plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)

    # Calculate the average confidence score where the model predicted this class
    predicted_confidence = confidence_map[predicted_mask]
    avg_confidence = np.mean(predicted_confidence) if predicted_confidence.size > 0 else 0.0

    # Overlay the average confidence score on top of the heatmap
    ax.text(
        0.5, 0.9, f"Mean_Conf: {avg_confidence:.2f}",
        color='white', fontsize=12, fontweight='bold', ha='center', va='center',
        transform=ax.transAxes
    )

    return heatmap


# class_names = ["Background", "Metal", "Glass", "Plastic", "Wood"]
def predict_and_visualize(model, image_path, device, weights_path, save_path, class_names):
    """
    Predict the segmentation mask for a single image and visualize the result.

    Args:
        model (nn.Module): The trained model.
        image_path (str): Path to the input image.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').
        transform (torchvision.transforms): The transformations to apply to the input image.
    """
    os.makedirs(save_path, exist_ok=True)
    if weights_path:
        model = load_pretrained_weights(model, weights_path, device)
    model.eval()  # Set the model to evaluation mode

    try:
        image = Image.open(image_path).convert("RGB")  # This will handle JPEG, PNG, etc.
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    # Load and transform the input image
    # image = Image.open(image_path).convert("RGB")
    transform = get_single_image_transform()
    input_image = transform(image).unsqueeze(0).to(device)

    print(f"Input image type: {type(input_image)}")
    print(f"Input image shape: {input_image.shape}")

    # Forward pass to get the prediction
    with torch.no_grad():
        output = model(input_image)
        print(f"Model output type: {type(output)}")
        print(f"Model output shape: {output.shape if isinstance(output, torch.Tensor) else 'Not a tensor'}")
    
    # Apply softmax to output to get class confidence scores
    output_softmax = F.softmax(output, dim=1).squeeze(0).cpu().numpy()
    predicted_class = np.argmax(output_softmax, axis=0)  # Predicted class for each pixel

    # Visualize the confidence maps for each class
    num_classes = output_softmax.shape[0]
    fig, axs = plt.subplots(2, 3, figsize=(15, 6))
    
    axs[0, 0].imshow(image)
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis("off")

    # For each class, plot the confidence map
    heatmaps = []
    for i in range(num_classes):
        row, col = divmod(i + 1, 3)
        heatmap = axs[row, col].imshow(output_softmax[i], cmap='hot', interpolation='nearest')
        axs[row, col].set_title(f"Class {class_names[i]} Confidence")
        axs[row, col].axis("off")

        # Overlay confidence score for each class on the image
        display_confidence_score(axs[row, col], output_softmax, predicted_class == i, material_name=class_names[i], class_index=i)
        heatmaps.append(heatmap)

    # Adjust the colorbar to be on the right side of the last column
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # [left, bottom, width, height] of the colorbar
    fig.colorbar(heatmaps[0], cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar on the right

    output_filename = os.path.join(save_path, os.path.basename(image_path).split('.')[0] + '_confidence_maps.png')
    plt.savefig(output_filename)
    plt.show()

# -----------------------------------------------------------------------------------------------------------------
def predict_and_visualize1(model, image_path, device, weights_path, save_path):
    """
    Predict the segmentation mask and visualize boundaries of materials.
    """
    os.makedirs(save_path, exist_ok=True)
    
    if weights_path:
        model = load_pretrained_weights(model, weights_path, device)
    model.eval()  # Set the model to evaluation mode

    try:
        image = Image.open(image_path).convert("RGB")  # Open the image
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    # Transform the input image
    transform = get_single_image_transform()
    input_image = transform(image).unsqueeze(0).to(device)

    print(f"Input image type: {type(input_image)}")
    print(f"Input image shape: {input_image.shape}")

    # Forward pass to get the prediction
    with torch.no_grad():
        output = model(input_image)
        print(f"Model output type: {type(output)}")
        print(f"Model output shape: {output.shape}")

    output_predictions = torch.argmax(output, 1).squeeze(0).cpu().numpy()

    # Decode the segmentation map to RGB
    decoded_predictions = decode_segmap(output_predictions, num_classes=5)  # Modify 'nc' to the number of classes you have

    # Find boundaries between classes
    boundaries = find_boundaries(output_predictions, mode='outer')

    # Overlay boundaries on the original image
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax[0, 0].imshow(image)
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis("off")

    # Overlay decoded prediction with boundaries
    ax[1].imshow(image)
    ax[1].imshow(decoded_predictions, alpha=0.6)  # Overlay segmentation map with some transparency
    ax[1].contour(boundaries, colors='red')  # Draw boundaries in red
    ax[1].set_title("Predicted Segmentation with Boundaries")
    ax[1].axis("off")

    output_filename = os.path.join(save_path, os.path.basename(image_path).split('.')[0] + '_segmentation_with_boundaries.png')
    plt.savefig(output_filename)
    plt.show()

# -------------------------------------------------------------------------------------------------------------------
def get_transforms(train=True, horizontal_flip_prob=0.0, rotation_degrees=0.0, resize=(256,256), crop_size=None):
    """
    Creates a torchvision transform pipeline for training and testing datasets. For training, augmentations
    such as horizontal flipping, random rotation, and random cropping can be included. For testing, only essential
    transformations like normalization and converting the image to a tensor are applied.

    Args:
        train (bool): Indicates whether the transform is for training or testing. If True, augmentations are applied.
        horizontal_flip_prob (float): Probability of applying a horizontal flip to the images. Effective only if train=True.
        rotation_degrees (float): The range of degrees for random rotation. Effective only if train=True.
        resize (tuple): The size to which the image will be resized.
        crop_size (tuple): The size of the crop for random cropping. Effective only if train=True.

    Returns:
        torchvision.transforms.Compose: Composed torchvision transforms for data preprocessing.
    """
    # Initialize an empty list to hold the transforms
    transform_list = []

    # Add transforms for both training and testing
    if train:
        if horizontal_flip_prob > 0.0:
            # Add random horizontal flip with the given probability
            transform_list.append(transforms.RandomHorizontalFlip(p=horizontal_flip_prob))
        
        if rotation_degrees > 0.0:
            # Add random rotation with the given range of degrees
            transform_list.append(transforms.RandomRotation(degrees=rotation_degrees))
        
        if crop_size:
            # Add random crop with the given size
            transform_list.append(transforms.RandomCrop(size=crop_size))

    # Resize images
    transform_list.append(transforms.Resize(resize))
    
    # Convert the image to a tensor
    transform_list.append(transforms.ToTensor())

    # Normalize the pixel values of the image
    # Normalized mean and std are derived from the ImageNet dataset and considered to be a well-established starting point
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    # Create a composed transform with the specified transforms
    composed_transform = transforms.Compose(transform_list)

    return composed_transform

def get_target_transform(resize = (256,256)):
    return transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

def get_single_image_transform():
    return transforms.Compose([
        # transforms.CenterCrop(256),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def write_results_to_csv(file_path, train_losses, test_losses, test_ious):
    """
    Writes the training and testing results to a CSV file.

    Args:
        file_path (str): Path to the CSV file where results will be saved. Without the postfix .csv
        train_losses (list): List of training losses.
        test_losses (list): List of testing losses.
        test_accuracies (list): List of testing accuracies.
    """
    os.makedirs(file_path, exist_ok=True)

    with open(file_path + ".csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Test IoU'])
        for epoch in range(len(train_losses)):
            writer.writerow([epoch + 1, train_losses[epoch], test_losses[epoch], test_ious[epoch]])

def plot_multiple_losses_and_accuracies(model_data_list):
    """
    Plots training and testing losses and accuracies for multiple models.

    Args:
        model_data_list (list of dict): A list of dictionaries containing the following keys:
            - 'name' (str): The name of the model (for the legend)
            - 'train_losses' (list): Training losses per epoch
            - 'test_losses' (list): Testing losses per epoch
            - 'test_accuracies' (list): Testing accuracies per epoch
    """
    # Plotting losses
    plt.figure(figsize=(10, 5))
    for model_data in model_data_list:
        plt.plot(model_data['train_losses'], label=f"{model_data['name']} Train Loss")
        plt.plot(model_data['test_losses'], label=f"{model_data['name']} Test Loss")

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Losses')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting accuracies
    plt.figure(figsize=(10, 5))
    for model_data in model_data_list:
        plt.plot(model_data['test_accuracies'], label=f"{model_data['name']} Test Accuracy")

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    pass # ToDo

def plot_samples_with_predictions(images, labels, predictions, class_names):
    """
    Plots a grid of images with labels and predictions, with dynamically adjusted text placement.

    Args:
        images (Tensor): Batch of images.
        labels (Tensor): True labels corresponding to the images.
        predictions (Tensor): Predicted labels for the images.
        class_names (list): List of class names indexed according to labels.
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.imshow(images[i].permute(1, 2, 0))
        ax.set_title(f"True: {class_names[labels[i]]}\nPredicted: {class_names[predictions[i]]}", fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    pass  # ToDo


def plot_confusion_matrix(labels, preds, class_names):
    """
    Plots a confusion matrix using ground truth labels and predictions.
    """
    cm = confusion_matrix(labels, preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt=".2f", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    pass  # ToDo

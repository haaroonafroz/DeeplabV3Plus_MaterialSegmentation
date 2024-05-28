import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ToDo import functions
def cross_entropy_4d(input, target):
    """
    Custom cross-entropy loss function for segmentation that supports 4D targets.
    
    Args:
        input (Tensor): Predictions from the model of shape [batch_size, num_classes, height, width].
        target (Tensor): Ground truth labels of shape [batch_size, height, width].
    
    Returns:
        Tensor: Computed loss.
    """
    print(f"Input shape: {input.shape}, Target shape: {target.shape}")

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

    return F.cross_entropy(input, target)

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, input, target):
        # Ensure that both input and target have the same shape
        assert input.shape == target.shape, "Input and target shapes must match"
        
        # Calculate the loss using your custom logic
        # Here we provide a simple example of mean squared error (MSE) loss
        loss = torch.mean((input - target)**2)
        
        return loss

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
    # Load the state_dict from the saved file
    state_dict = torch.load(weights_path, map_location=device)
    
    # Load the state_dict into the network
    network.load_state_dict(state_dict)
    
    return network
    pass # ToDo


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
    pass  # ToDo

def save_model(model, path):
    """
    Saves the model state_dict to a specified file.

    Args:
        model (nn.Module): The PyTorch model to save. Only the state_dict should be saved.
        path (str): The path where to save the model. Without the postifix .pth
    """
    # Ensure device independence
    model_state = model.to(torch.device('cpu')).state_dict()
    torch.save(model_state, path + ".pth")
    pass  # ToDo

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
    pass  # ToDo

def get_transforms(train=True, horizontal_flip_prob=0.0, rotation_degrees=0.0, resize=(375, 500)):
    """
    Creates a torchvision transform pipeline for training and testing datasets. For training, augmentations
    such as horizontal flipping and random rotation can be included. For testing, only essential transformations
    like normalization and converting the image to a tensor are applied.

    Args:
        train (bool): Indicates whether the transform is for training or testing. If True, augmentations are applied.
        horizontal_flip_prob (float): Probability of applying a horizontal flip to the images. Effective only if train=True.
        rotation_degrees (float): The range of degrees for random rotation. Effective only if train=True.

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

    # Resize images
    transform_list.append(transforms.Resize(resize))
    
    # Convert the image to a tensor
    transform_list.append(transforms.ToTensor())

    # Normalize the pixel values of the image
    # Normalized mean and std are derived from the ImageNet dataset and considered to be a well established starting point
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    # Create a composed transform with the specified transforms
    composed_transform = transforms.Compose(transform_list)

    return composed_transform
    #pass  # ToDo
def get_target_transform():
    return transforms.Compose([
        transforms.Resize((375, 500)),
        transforms.ToTensor()
    ])

def write_results_to_csv(file_path, train_losses, test_losses, test_accuracies):
    """
    Writes the training and testing results to a CSV file.

    Args:
        file_path (str): Path to the CSV file where results will be saved. Without the postfix .csv
        train_losses (list): List of training losses.
        test_losses (list): List of testing losses.
        test_accuracies (list): List of testing accuracies.
    """
    with open(file_path + ".csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Test Accuracy'])
        for epoch in range(len(train_losses)):
            writer.writerow([epoch + 1, train_losses[epoch], test_losses[epoch], test_accuracies[epoch]])

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

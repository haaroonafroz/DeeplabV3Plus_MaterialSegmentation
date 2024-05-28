import torch
import torch.nn.functional as F
from tqdm import tqdm
from dlcv.utils import cross_entropy_4d

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    """
    Trains a given model for one epoch using the provided data loader, criterion, and optimizer.

    Args:
        model (nn.Module): The model to be trained.
        data_loader (DataLoader): The data loader providing the training data.
        criterion (nn.Module): The loss function to be used during training.
        optimizer (torch.optim.Optimizer): The optimizer to be used for updating the model's parameters.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').

    Returns:
        float: The average loss per batch for the entire epoch.
    """
    model.train()  # Set the model to training mode
    epoch_loss = 0.0

    # Iterate over the data loader
    for inputs, targets in tqdm(data_loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to the appropriate device
        optimizer.zero_grad()  # Zero out gradients

        # Forward pass
        outputs = model(inputs)
        loss = cross_entropy_4d(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()  # Update model parameters
        
        epoch_loss += loss.item() * inputs.size(0)  # Accumulate loss

    epoch_loss /= len(data_loader.dataset)  # Calculate average loss
    return epoch_loss

def evaluate_one_epoch(model, data_loader, device):
    """
    Tests a given model for one epoch using the provided data loader and criterion.

    Args:
        model (nn.Module): The model to be tested.
        data_loader (DataLoader): The data loader providing the testing data.
        criterion (nn.Module): The loss function to be used during testing.
        device (torch.device): The device on which the model is running (e.g., 'cpu' or 'cuda').

    Returns:
        float: The average loss per batch for the entire epoch.
        float: The accuracy of the model on the test data.
    """
    model.eval()  # Set the model to evaluation mode
    epoch_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # Iterate over the data loader
    with torch.no_grad():  # Disable gradient calculation during evaluation
        for inputs, targets in tqdm(data_loader, desc="Evaluation"):
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the appropriate device
            
            # If target has a channel dimension, squeeze it
            if targets.ndimension() == 4 and targets.size(1) == 1:
                targets = targets.squeeze(1)

            # Ensure the target is of type Long
            targets = targets.long()
            # Forward pass
            outputs = model(inputs)
            loss = cross_entropy_4d(outputs, targets)
        
            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_predictions += targets.size(0)

            epoch_loss += loss.item() * inputs.size(0)  # Accumulate loss

    epoch_loss /= len(data_loader.dataset)  # Calculate average loss
    accuracy = correct_predictions / total_predictions  # Calculate accuracy

    return epoch_loss, accuracy

def train_and_evaluate_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, scheduler=None, early_stopping=False):
    train_losses = []
    test_losses = []
    test_accuracies = []
    best_test_loss = float('inf')
    consecutive_no_improvement = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        test_loss, test_accuracy = evaluate_one_epoch(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

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

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return train_losses, test_losses, test_accuracies


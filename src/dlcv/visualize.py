import os
import csv
import matplotlib.pyplot as plt

cwd = os.getcwd()

def read_csv_results(file_path):
    """
    Reads the CSV file and returns lists for epoch, train loss, test loss, and test IoU.

    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        epochs (list), train_losses (list), test_losses (list), test_ious (list)
    """
    epochs, train_losses, test_losses, test_ious = [], [], [], []

    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            epochs.append(int(row['Epoch']))
            train_losses.append(float(row['Train Loss']))
            test_losses.append(float(row['Test Loss']))
            test_ious.append(float(row['Test IoU']))

    return epochs, train_losses, test_losses, test_ious

# Manually specify the files in order
csv_file_0_50 = cwd + '/src/dlcv/results/Semantic_Segmentation_0-50Epochs_NewDataset.csv'  
csv_file_51_100 = cwd + '/src/dlcv/results/Semantic_Segmentation_50-100Epochs_NewDataset.csv'  

# Read results from both CSV files
combined_epochs, combined_train_losses, combined_test_losses, combined_test_ious = [], [], [], []

# First file (Epochs 0-50)
epochs, train_losses, test_losses, test_ious = read_csv_results(csv_file_0_50)
combined_epochs.extend(epochs)
combined_train_losses.extend(train_losses)
combined_test_losses.extend(test_losses)
combined_test_ious.extend(test_ious)

# Second file (Epochs 51-100)
epochs, train_losses, test_losses, test_ious = read_csv_results(csv_file_51_100)
# Shift epoch numbers for continuation
last_epoch = combined_epochs[-1]
epochs = [epoch + last_epoch for epoch in epochs]
combined_epochs.extend(epochs)
combined_train_losses.extend(train_losses)
combined_test_losses.extend(test_losses)
combined_test_ious.extend(test_ious)

# Plot each metric individually

# Plot Train Loss
plt.figure(figsize=(10, 6))
plt.plot(combined_epochs, combined_train_losses, label='Train Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Train Loss')
plt.title('Training Loss- DeeplabV3+ with Mobilenet')
plt.ylim([-2, 2])
plt.legend(loc='best')
plt.show()

# Plot Test Loss
plt.figure(figsize=(10, 6))
plt.plot(combined_epochs, combined_test_losses, label='Test Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Test Loss')
plt.title('Test Loss- DeeplabV3+ with Mobilenet')
plt.ylim([-2, 2])
plt.legend(loc='best')
plt.show()

# Plot Test IoU
plt.figure(figsize=(10, 6))
plt.plot(combined_epochs, combined_test_ious, label='Test IoU', color='green')
plt.xlabel('Epochs')
plt.ylabel('Test IoU')
plt.title('Material_IoU- DeeplabV3+ with Mobilenet')
plt.legend(loc='best')
plt.show()
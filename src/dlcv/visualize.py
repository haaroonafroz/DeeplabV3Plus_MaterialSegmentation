import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# This package internal functions should be used here
from torchvision.datasets import Food101
from src.dlcv.model import CustomConvNeXt
from src.dlcv.utils import load_pretrained_weights, plot_samples_with_predictions, plot_confusion_matrix, get_transforms
from src.dlcv.dataset import SubsetSTL10

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    # Load the model
    model = CustomConvNeXt(in_channels=3, num_classes=10, num_blocks=[3, 3, 9, 3], hidden_dims=[96, 192, 384, 768]) #STL10
    #model = CustomConvNeXt(in_channels=3, num_classes=101, num_blocks=[3, 3, 9, 3], hidden_dims=[96, 192, 384, 768]) #Food101

    model = load_pretrained_weights(model, args.pretrained_weights, device)

    # Load the data
    transform = get_transforms(train=False)

    dataset = SubsetSTL10(root=args.data_root, split='test', transform=transform, download=True)
    #dataset = test_dataset = Food101(root='/kaggle/input/food101/food-101', split='test', download=False, transform=test_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    all_preds = []
    all_labels = []
    plotted = False
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if not plotted:
                plot_samples_with_predictions(images.cpu(), labels.cpu(), preds.cpu(), dataset.classes) # ToDo <- add this function in utils.py
                plotted = True  # Set flag to True after plotting once

    plot_confusion_matrix(all_labels, all_preds, dataset.classes) # ToDo <- add this function in utils.py


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on a trained model and plot results.')
    parser.add_argument('--pretrained_weights', type=str, help='Path to pretrained weights file')
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory for dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    #parser.add_argument('--conv_layers', type=int, default=2, help='Number of convolutional layers', choices=[2,3])
    #parser.add_argument('--filters_conv1', type=int, default=16, help='Number of filters in the first conv layer')
    #parser.add_argument('--filters_conv2', type=int, default=32, help='Number of filters in the second conv layer')
    #parser.add_argument('--filters_conv3', type=int, required=False, help='Number of filters in the third conv layer')
    #parser.add_argument('--dense_units', type=int, default=128, help='Number of units in the dense layer')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even if available')

    args = parser.parse_args()
    main(args)

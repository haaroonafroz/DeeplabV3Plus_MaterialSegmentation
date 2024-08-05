import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
from yacs.config import CfgNode as CN
from dlcv.config import get_cfg_defaults, get_cfg_from_file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# This package internal functions should be used here
from torchvision.datasets import VOCSegmentation
from dlcv.model import DeepLabV3Model
from dlcv.utils import * #load_pretrained_weights, plot_samples_with_predictions, plot_confusion_matrix, get_transforms

def main(args):

    cfg = get_cfg_from_file(args.config_file)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    # Load the model
    #weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
    model = DeepLabV3Model(num_classes=cfg.MODEL.NUM_CLASSES)
    model.to(device)

    # Load the data
    transform = get_transforms(train=False)
    target_transform = get_target_transform()

    dataset = VOCSegmentation(root=cfg.DATA.ROOT, year='2012', image_set='val', download=False, transform=transform, target_transform=target_transform)
    #dataset = test_dataset = Food101(root='/kaggle/input/food101/food-101', split='test', download=False, transform=test_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

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
    parser.add_argument('--config_file', type=str, required=True, help='Path to the config file used during training')
    parser.add_argument('--saved_model_path', type=str, required=True, help='Path to the saved model weights file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA even if available')

    args = parser.parse_args()
    main(args)

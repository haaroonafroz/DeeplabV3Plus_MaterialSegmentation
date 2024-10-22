import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import sys
import os
from yacs.config import CfgNode as CN
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# This package internal functions should be used here
from dlcv.config import get_cfg_defaults
from dlcv.model import *
from dlcv.utils import *
from dlcv.training import *
from dlcv.dataset import *
from dlcv.training_unet import *

class_names = ["background", "Metal", "Glass", "Plastic", "Wood"]
def sum(a, b):
    x=a+b
    return x

def main(cfg, mode, image_path=None):
    print(f"Using configuration file: {config_file_path}")
    print("Configuration for this run:")
    print(cfg.dump())

    if torch.cuda.is_available() and not cfg.MISC.NO_CUDA:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # torch.backends.cudnn.benchmark = False  # Disable cuDNN benchmarking
    # torch.backends.cudnn.deterministic = True

    train_transform = get_transforms(
                                    train=True,
                                    horizontal_flip_prob=cfg.AUGMENTATION.HORIZONTAL_FLIP_PROB,
                                    rotation_degrees=cfg.AUGMENTATION.ROTATION_DEGREES,
                                    crop_size=cfg.AUGMENTATION.CROP_SIZE
                                    )
    test_transform = get_transforms(train=False)
    target_transform = get_target_transform()

    # Model selection based on configuration
    if cfg.MODEL.MODEL == 'Deeplabv3Plus':
        model = get_model(num_classes_material=cfg.MODEL.NUM_CLASSES_MATERIAL, backbone=cfg.MODEL.BACKBONE)
    elif cfg.MODEL.MODEL == 'U-Net':
        model = get_model_unet(num_classes=cfg.MODEL.NUM_CLASSES_MATERIAL)
    else:
        raise ValueError(f"Model {cfg.MODEL.MODEL} not supported. Choose 'Deeplabv3Plus' or 'U-Net'.")

    model.to(device)

    if mode == 'single_image' and image_path:
        if image_path is None:
            raise ValueError("Image path must be provided for single image mode")

        predict_and_visualize_with_edges(model, image_path, device, weights_path= cfg.MISC.PRETRAINED_WEIGHTS, save_path=cfg.MISC.SAVE_PREDICTION, class_names=class_names)
        predict_and_visualize(model, image_path, device, weights_path= cfg.MISC.PRETRAINED_WEIGHTS, save_path=cfg.MISC.SAVE_PREDICTION, class_names=class_names)


    else:
        with open(os.path.join(cfg.DATA.MATERIAL_ROOT, 'annotations.json'), 'r') as f:
            annotations = json.load(f)['annotations']

        # Split the annotations into training and testing sets
        train_annotations, test_annotations = train_test_split(annotations, test_size=0.2, random_state=42)

        # load dataset
        train_dataset = CustomSegmentationDataset(
            material_root=cfg.DATA.MATERIAL_ROOT,
            transform=train_transform,
            target_transform=target_transform,
            mode='train',
            annotations=train_annotations
        )
        test_dataset = CustomSegmentationDataset(
            material_root=cfg.DATA.MATERIAL_ROOT,
            transform=test_transform,
            target_transform=target_transform,
            mode='test',
            annotations=test_annotations
        )

        train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=2)

        if cfg.MISC.PRETRAINED_WEIGHTS:
            model = load_pretrained_weights(model, cfg.MISC.PRETRAINED_WEIGHTS, device)

        # Training mode
        if mode == 'train':
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.BASE_LR)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA)

            if cfg.MODEL.MODEL == 'Deeplabv3Plus':
                # Train Deeplabv3+ model
                train_losses, test_losses, test_ious_material = train_and_evaluate_model(
                    model, train_loader, test_loader, combined_loss, optimizer, 
                    cfg.TRAIN.NUM_EPOCHS, device, scheduler=scheduler, early_stopping=cfg.TRAIN.EARLY_STOPPING
                )
            elif cfg.MODEL.MODEL == 'U-Net':
                # Train U-Net model
                train_losses, test_losses, test_ious_material = train_and_evaluate_model_unet(
                    model, train_loader, test_loader, combined_loss, optimizer, 
                    cfg.TRAIN.NUM_EPOCHS, device, scheduler=scheduler, early_stopping=cfg.TRAIN.EARLY_STOPPING
                )

            print(f"Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Test IoU (Material): {test_ious_material[-1]:.4f}")
            write_results_to_csv(cfg.MISC.RESULTS_CSV + "/" + cfg.MISC.RUN_NAME, train_losses, test_losses, test_ious_material)

            if cfg.MISC.SAVE_MODEL_PATH:
                save_model(model, cfg.MISC.SAVE_MODEL_PATH + "/" + cfg.MISC.RUN_NAME)

            # Save config used in this run
            config_save_path = os.path.join(cfg.MISC.SAVE_MODEL_PATH, cfg.MISC.RUN_NAME + '_runConfig.yaml')
            with open(config_save_path, 'w') as f:
                f.write(cfg.dump())

        # Test mode
        elif mode == 'test':
            if cfg.MODEL.MODEL == 'Deeplabv3Plus':
                test_loss, test_iou_material = evaluate_one_epoch(
                    model, test_loader, device, criterion=combined_loss
                )
            elif cfg.MODEL.MODEL == 'U-Net':
                test_loss, test_iou_material = evaluate_one_epoch_unet(
                    model, test_loader, device, criterion=combined_loss
                )

            print(f"Test Loss: {test_loss:.4f}, Test IoU (Material): {test_iou_material:.4f}")
            write_results_to_csv(cfg.MISC.RESULTS_CSV + "/" + cfg.MISC.RUN_NAME, [test_loss], [test_iou_material])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and Evaluate a model")
    parser.add_argument('--config', type=str, help="Path to the config file")
    parser.add_argument('--image_path', type=str, help="Path to the input image for 'single_image' mode")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test', 'single_image'], help="Mode to run the script in: 'train' or 'test'")
    args = parser.parse_args()
    
    cfg = get_cfg_defaults()

    # Check if the script is run with the command-line argument or from the notebook
    if args.config:
        config_file_path = args.config
    else:
        # Fallback to an environment variable or a default config path
        config_file_path = os.getenv('CONFIG_FILE', 'configs/kaggle_test1.yaml')

    cfg.CONFIG_FILE_PATH = config_file_path
    cfg.merge_from_file(config_file_path)
    main(cfg, args.mode, args.image_path)
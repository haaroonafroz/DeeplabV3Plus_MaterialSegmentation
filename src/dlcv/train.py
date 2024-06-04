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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# This package internal functions should be used here
from dlcv.config import get_cfg_defaults
#from src.dlcv.dataset import SubsetSTL10
from dlcv.model import DeepLabV3Model
from dlcv.utils import *
from dlcv.training import *
#ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def sum(a, b):
    x=a+b
    return x

def main(cfg, mode):
    print(f"Using configuration file: {config_file_path}")
    print("Configuration for this run:")
    print(cfg.dump())

    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.MISC.NO_CUDA else "cpu")
    print(f"Device: {device}")

    train_transform = get_transforms(train=True, horizontal_flip_prob=cfg.AUGMENTATION.HORIZONTAL_FLIP_PROB, rotation_degrees=cfg.AUGMENTATION.ROTATION_DEGREES)
    test_transform = get_transforms(train=False)
    target_transform = get_target_transform()

    train_dataset = VOCSegmentation(root=cfg.DATA.ROOT, year='2012', image_set='train', download=True, transform=train_transform, target_transform=target_transform)
    test_dataset = VOCSegmentation(root=cfg.DATA.ROOT, year='2012', image_set='val', download=True, transform=test_transform, target_transform=target_transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=4)

    #weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
    model = DeepLabV3Model(num_classes=cfg.MODEL.NUM_CLASSES)
    model.to(device)

    if cfg.MISC.PRETRAINED_WEIGHTS:
        model = load_pretrained_weights(model, cfg.MISC.PRETRAINED_WEIGHTS, device)

    # if cfg.MISC.FROZEN_LAYERS:
    #     freeze_layers(model, cfg.MISC.FROZEN_LAYERS)

    # optimizer = Adam(model.parameters(), lr=cfg.TRAIN.BASE_LR)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA)

    # train_losses, test_losses, test_accuracies = train_and_evaluate_model(model, train_loader, test_loader,
    #                                                                        cross_entropy_4d, optimizer,
    #                                                                     cfg.TRAIN.NUM_EPOCHS, device,
    #                                                                     scheduler=scheduler,
    #                                                                     early_stopping=cfg.TRAIN.EARLY_STOPPING
    # )

    # write_results_to_csv(cfg.MISC.RESULTS_CSV + "/" + cfg.MISC.RUN_NAME, train_losses, test_losses, test_accuracies)

    # if cfg.MISC.SAVE_MODEL_PATH:
    #     save_model(model, cfg.MISC.SAVE_MODEL_PATH + "/" + cfg.MISC.RUN_NAME)

    # config_save_path = os.path.join(cfg.MISC.SAVE_MODEL_PATH, cfg.MISC.RUN_NAME + '_runConfig.yaml')
    # with open(config_save_path, 'w') as f:
    #     f.write(cfg.dump())

# -------------------------------------------------------------------------------------------------------

    if mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.BASE_LR)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA)

        train_losses, test_losses, test_accuracies = train_and_evaluate_model(
            model, train_loader, test_loader, cross_entropy_4d, optimizer, cfg.TRAIN.NUM_EPOCHS, device, scheduler=scheduler,
            early_stopping=cfg.TRAIN.EARLY_STOPPING)

        write_results_to_csv(cfg.MISC.RESULTS_CSV + "/" + cfg.MISC.RUN_NAME, train_losses, test_losses, test_accuracies)

        if cfg.MISC.SAVE_MODEL_PATH:
            save_model(model, cfg.MISC.SAVE_MODEL_PATH + "/" + cfg.MISC.RUN_NAME + ".pth")

        config_save_path = os.path.join(cfg.MISC.SAVE_MODEL_PATH, cfg.MISC.RUN_NAME + '_runConfig.yaml')
        with open(config_save_path, 'w') as f:
            f.write(cfg.dump())

    elif mode == 'test':
        test_loss, test_accuracy = evaluate_one_epoch(model, test_loader, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and Evaluate a model")
    parser.add_argument('--config', type=str, help="Path to the config file")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help="Mode to run the script in: 'train' or 'test'")
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
    main(cfg, args.mode)
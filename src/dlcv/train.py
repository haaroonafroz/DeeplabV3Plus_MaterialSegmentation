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
#from torchvision.datasets import Food101
import torchvision.transforms as transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# This package internal functions should be used here
from dlcv.config import get_cfg_defaults
#from src.dlcv.dataset import SubsetSTL10
from dlcv.model import DeepLabV3Model
from dlcv.utils import *
from dlcv.training import train_and_evaluate_model

#ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def sum(a, b):
    x=a+b
    return x

def main(cfg):#, stratification_rates):

    print(f"Using configuration file: {config_file_path}")
    print("Configuration for this run:")
    print(cfg.dump())

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.MISC.NO_CUDA else "cpu")
    # ToDo

    # Define transformations for training and testing
    train_transform = get_transforms(train=True, horizontal_flip_prob=cfg.AUGMENTATION.HORIZONTAL_FLIP_PROB, rotation_degrees=cfg.AUGMENTATION.ROTATION_DEGREES)
    test_transform = get_transforms(train=False)
    target_transform = get_target_transform()
    # ToDo

    # Load datasets
    #train_dataset = SubsetSTL10(root=cfg.DATA.ROOT, split='train', download=True, transform=train_transform)
    #test_dataset = SubsetSTL10(root=cfg.DATA.ROOT, split='test', download=True, transform=test_transform)

    train_dataset = VOCSegmentation(root=cfg.DATA.ROOT, year='2012', image_set='train', download=True, transform=train_transform, target_transform=target_transform)
    test_dataset = VOCSegmentation(root=cfg.DATA.ROOT, year='2012', image_set='val', download=True, transform=test_transform, target_transform=target_transform)
    # ToDo

    # Initialize training and test loader
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=4)
    # ToDo

    # Initialize model
    #model = CustomConvNeXt(in_channels=3, num_classes=10, num_blocks=cfg.MODEL.NUM_BLOCKS, hidden_dims=cfg.MODEL.HIDDEN_DIMS) #STL10
    model = DeepLabV3Model(num_classes=cfg.MODEL.NUM_CLASSES)
    model.to(device)

    # Load pretrained weights if specified
    if cfg.MISC.PRETRAINED_WEIGHTS:
        model = load_pretrained_weights(model, cfg.MISC.PRETRAINED_WEIGHTS, device)
    # ToDo

    # Freeze layers if set as argument
    if cfg.MISC.FROZEN_LAYERS:
        freeze_layers(model, cfg.MISC.FROZEN_LAYERS)
    # ToDo

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.BASE_LR)
    # ToDo

    # Define the criterion
    #criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    # ToDo

    # Define a scheduler - use the MultiStepLR scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA)
    # ToDo

    # Hand everything to the train and evaluate model function
    train_losses, test_losses, test_accuracies = train_and_evaluate_model(model, train_loader, test_loader,
                                                                          cross_entropy_4d, optimizer, cfg.TRAIN.NUM_EPOCHS,
                                                                          device, scheduler=scheduler,
                                                                          early_stopping=cfg.TRAIN.EARLY_STOPPING)
    # ToDo

    # Save results to CSV
    write_results_to_csv(cfg.MISC.RESULTS_CSV + "/" + cfg.MISC.RUN_NAME, train_losses, test_losses, test_accuracies)

    # Save the model using the default folder
    if cfg.MISC.SAVE_MODEL_PATH:
        save_model(model, cfg.MISC.SAVE_MODEL_PATH + "/" + cfg.MISC.RUN_NAME + ".pth")

    config_save_path = os.path.join(cfg.MISC.SAVE_MODEL_PATH, cfg.MISC.RUN_NAME + '_runConfig.yaml')
    with open(config_save_path, 'w') as f:
        f.write(cfg.dump())

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    # Assuming the config file is passed as an environment variable
    config_file_path = os.getenv('CONFIG_FILE', 'config/config1.yaml')
    cfg.merge_from_file(config_file_path)
    main(cfg)
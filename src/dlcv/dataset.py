import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F

class CustomSegmentationDataset(Dataset):
    def __init__(self, material_root, transform=None, target_transform=None, mode='train', annotations=None):
        self.material_root = material_root
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        
        # Load Material dataset annotations
        if annotations is None:
            with open(os.path.join(material_root, 'annotations.json'), 'r') as f:
                self.material_annotations = json.load(f)['annotations']
        else:
            self.material_annotations = annotations

        # Load Material dataset images and masks
        material_image_dir = os.path.join(material_root, 'JPEGImages')
        material_binary_mask_dir = os.path.join(material_root, 'Binary_Masks')
        material_class_mask_dir = os.path.join(material_root, 'Class_Masks')

        # Create a dictionary to hold lists of images for each class
        self.class_images = {
            'glass': [],
            'plastic': [],
            'wood': [],
            'metal': [],
            'mixed': []
        }

        # Populate the class_images dictionary based on file naming conventions
        for ann in self.material_annotations:
            image_name = ann['image']
            if '_g' in image_name:
                self.class_images['glass'].append(ann)
            elif '_p' in image_name:
                self.class_images['plastic'].append(ann)
            elif '_w' in image_name:
                self.class_images['wood'].append(ann)
            elif '_m' in image_name:
                self.class_images['metal'].append(ann)
            elif '_mx' in image_name:
                self.class_images['mixed'].append(ann)

        # Split the dataset into train and test sets with equal representation
        self.train_annotations, self.test_annotations = self.split_balanced_dataset()

    def split_balanced_dataset(self, test_size=0.2):
        train_annotations = []
        test_annotations = []

        for class_name, annotations in self.class_images.items():
            # Shuffle the annotations for randomness
            random.shuffle(annotations)

            # Determine the split point
            split_idx = int(len(annotations) * (1 - test_size))

            # Append the images to train and test sets
            train_annotations.extend(annotations[:split_idx])
            test_annotations.extend(annotations[split_idx:])

        return train_annotations, test_annotations

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_annotations)
        else:
            return len(self.test_annotations)

    def __getitem__(self, idx):
        # Use train or test annotations based on mode
        if self.mode == 'train':
            ann = self.train_annotations[idx]
        else:
            ann = self.test_annotations[idx]

        # Get item from Material dataset
        image = Image.open(os.path.join(self.material_root, 'JPEGImages', ann['image'])).convert("RGB")
        binary_mask = Image.open(os.path.join(self.material_root, 'Binary_Masks', ann['binary_mask'])).convert("L")
        class_mask = Image.open(os.path.join(self.material_root, 'Class_Masks', ann['class_mask'])).convert("L")

        # Resize
        image = image.resize((256, 256), Image.BILINEAR)
        binary_mask = binary_mask.resize((256, 256), Image.NEAREST)
        class_mask = class_mask.resize((256, 256), Image.NEAREST)

        # Apply the appropriate transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            binary_mask = self.target_transform(binary_mask)
            class_mask = self.target_transform(class_mask)

        return image, binary_mask, class_mask

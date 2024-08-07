import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomSegmentationDataset(Dataset):
    def __init__(self, material_root, transform=None, target_transform=None, mode='train'):
        self.material_root = material_root
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

        # Load Material dataset annotations
        with open(os.path.join(material_root, 'annotations.json'), 'r') as f:
            self.material_annotations = json.load(f)['annotations']

        # Load Material dataset images and masks
        material_image_dir = os.path.join(material_root, 'JPEGImages')
        material_binary_mask_dir = os.path.join(material_root, 'Binary_Masks')
        material_class_mask_dir = os.path.join(material_root, 'Class_Masks')

        self.material_images = [os.path.join(material_image_dir, ann['image']) for ann in self.material_annotations]
        self.material_binary_masks = [os.path.join(material_binary_mask_dir, ann['binary_mask']) for ann in self.material_annotations]
        self.material_class_masks = [os.path.join(material_class_mask_dir, ann['class_mask']) for ann in self.material_annotations]

    def __len__(self):
        return len(self.material_annotations)

    def __getitem__(self, idx):
        # Get item from Material dataset
        image = Image.open(self.material_images[idx]).convert("RGB")
        binary_mask = Image.open(self.material_binary_masks[idx]).convert("L")
        class_mask = Image.open(self.material_class_masks[idx]).convert("L")

        # Resize the class mask
        # Resize images and masks
        image = image.resize((375, 375), Image.BILINEAR)
        binary_mask = binary_mask.resize((375, 375), Image.NEAREST)
        class_mask = class_mask.resize((375, 375), Image.NEAREST)

        # Apply the appropriate transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            binary_mask = self.target_transform(binary_mask)
            class_mask = self.target_transform(class_mask)

        return image, binary_mask, class_mask

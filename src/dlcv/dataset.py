import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F

class CustomSegmentationDataset(Dataset):
    def __init__(self, material_root, transform=None, target_transform=None, mode='train', downsample_size=2):
        self.material_root = material_root
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.downsample_size = downsample_size

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

        # Resize
        image = image.resize((256,256), Image.BILINEAR)
        binary_mask = binary_mask.resize((256,256), Image.NEAREST)
        class_mask = class_mask.resize((256,256), Image.NEAREST)

        # Downsample the class mask using max pooling
        class_mask_tensor = torch.from_numpy(np.array(class_mask)).float()
        class_mask_downsampled = max_pooling_downsample(class_mask_tensor, self.downsample_size)

        # Apply the appropriate transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            binary_mask = self.target_transform(binary_mask)
            class_mask_downsampled = self.target_transform(class_mask_downsampled)

        return image, binary_mask, class_mask_downsampled
    
def max_pooling_downsample(class_mask, pool_size):
    class_mask = class_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    downsampled_mask = F.max_pool2d(class_mask, kernel_size=pool_size, stride=pool_size)
    return downsampled_mask.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

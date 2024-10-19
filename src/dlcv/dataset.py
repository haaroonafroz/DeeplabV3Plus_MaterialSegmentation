import os
import json
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
        material_class_mask_dir = os.path.join(material_root, 'Class_Labels')

        self.material_images = [os.path.join(material_image_dir, ann['image']) for ann in self.material_annotations]
        self.material_binary_masks = [os.path.join(material_binary_mask_dir, ann['binary_mask']) for ann in self.material_annotations]
        self.material_class_masks = [os.path.join(material_class_mask_dir, ann['class_mask']) for ann in self.material_annotations]

    def __len__(self):
        return len(self.material_annotations)

    def __getitem__(self, idx):
        # Get item from Material dataset
        image = Image.open(self.material_images[idx]).convert("RGB")
        binary_mask = Image.open(self.material_binary_masks[idx]).convert("L")  # Convert to grayscale (L mode)

        # Convert the binary mask to 0 and 1 by thresholding the grayscale values
        binary_mask = np.array(binary_mask)  # Convert to numpy array
        binary_mask = np.where(binary_mask > 128, 1, 0).astype(np.uint8)  # Threshold: 255 becomes 1, 0 stays 0

        # Load the class mask as an .npy file
        class_mask = np.load(self.material_class_masks[idx])

        # Resize the image and binary mask
        image = image.resize((256,256), Image.BILINEAR)
        
        # No need to convert binary mask here as it has been thresholded already
        binary_mask = Image.fromarray(binary_mask).resize((256,256), Image.NEAREST)
        binary_mask = torch.from_numpy(np.array(binary_mask)).long()  # Convert to torch tensor

        # Resize class mask (from .npy file)
        class_mask = Image.fromarray(class_mask)  # Convert npy array to PIL Image for resizing
        class_mask = class_mask.resize((256, 256), Image.NEAREST)  # Resize with nearest-neighbor interpolation
        class_mask = np.array(class_mask)  # Convert back to numpy array
        class_mask = torch.from_numpy(class_mask).long()  # Convert the resized npy mask to a torch tensor

        # Apply the appropriate transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            binary_mask = self.target_transform(binary_mask)
        
        return image, binary_mask, class_mask

    
def max_pooling_downsample(class_mask, pool_size):
    class_mask = class_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    downsampled_mask = F.max_pool2d(class_mask, kernel_size=pool_size, stride=pool_size)
    return downsampled_mask.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

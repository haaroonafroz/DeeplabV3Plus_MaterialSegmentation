import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import VOCSegmentation

class CustomSegmentationDataset(Dataset):
    def __init__(self, voc_root, material_root, train_transform=None, test_transform=None, target_transform=None, mode='train'):
        self.voc_root = voc_root
        self.material_root = material_root
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.target_transform = target_transform
        self.mode = mode

        # Load VOC dataset
        image_set = 'train' if mode == 'train' else 'val'
        self.voc_dataset = VOCSegmentation(root=voc_root, year='2012', image_set=image_set, download=True)

        # Load Material dataset annotations
        with open(os.path.join(material_root, 'annotations.json'), 'r') as f:
            self.material_annotations = json.load(f)['annotations']

        # Load Material dataset images and masks
        material_image_dir = os.path.join(material_root, 'JPEGImages')
        material_binary_mask_dir = os.path.join(material_root, 'BinaryMasks')
        material_class_mask_dir = os.path.join(material_root, 'ClassMasks')

        self.material_images = [os.path.join(material_image_dir, ann['image']) for ann in self.material_annotations]
        self.material_binary_masks = [os.path.join(material_binary_mask_dir, ann['binary_mask']) for ann in self.material_annotations]
        self.material_class_masks = [os.path.join(material_class_mask_dir, ann['class_mask']) for ann in self.material_annotations]

    def __len__(self):
        return len(self.voc_dataset) + len(self.material_annotations)

    def __getitem__(self, idx):
        if idx < len(self.voc_dataset):
            # Get item from VOC dataset
            image, mask = self.voc_dataset[idx]
            class_mask = Image.new("L", mask.size)  # Create an empty class mask for VOC images
        else:
            # Get item from Material dataset
            material_idx = idx - len(self.voc_dataset)
            image = Image.open(self.material_images[material_idx]).convert("RGB")
            mask = Image.open(self.material_binary_masks[material_idx]).convert("L")
            class_mask = Image.open(self.material_class_masks[material_idx]).convert("L")

        # Apply the appropriate transforms
        if self.mode == 'train' and self.train_transform:
            image = self.train_transform(image)
        elif self.test_transform:
            image = self.test_transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)
            class_mask = self.target_transform(class_mask)

        return image, mask, class_mask

# Define your transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images and masks to a fixed size
    transforms.ToTensor()
])

# Create dataset and dataloader
voc_root = './data/VOCdevkit/VOC2012'
material_root = './data/material_dataset'
dataset = CustomSegmentationDataset(voc_root, material_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# For evaluation, you can create a similar dataset but with the 'val' split for VOCSegmentation
eval_voc_root = './data/VOCdevkit/VOC2012'
eval_material_root = './data/material_dataset'
eval_dataset = CustomSegmentationDataset(eval_voc_root, eval_material_root, transform=transform)
eval_dataloader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=4)

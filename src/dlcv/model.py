import torch
import torch.nn as nn
import torchvision.models.segmentation as models
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

class DeepLabV3Model(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Model, self).__init__()
        # Load pre-trained Deeplabv3 model with a ResNet101 backbone
        weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        self.model = models.deeplabv3_resnet101(weights = weights)
        # Adjust the classifier to output the correct number of classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
    def forward(self, x):
        return self.model(x)['out']

def get_model(num_classes):
    return DeepLabV3Model(num_classes)

if __name__ == "__main__":
    num_classes = 21  # Example number of classes for PASCAL VOC 2012
    model = get_model(num_classes)
    print(model)
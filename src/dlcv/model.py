import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

import ssl
import certifi

# ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = ssl._create_unverified_context

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier_object, classifier_material):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier_object = classifier_object
        self.classifier_material = classifier_material
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x_object = self.classifier_object(features)
        x_material = self.classifier_material(features)
        x_object = F.interpolate(x_object, size=input_shape, mode='bilinear', align_corners=False)
        x_material = F.interpolate(x_material, size=input_shape, mode='bilinear', align_corners=False)
        return x_object, x_material

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class MobileNetV2Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV2Backbone, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        self.low_level_layers = mobilenet.features[:4]  # Up to layer 4 (inclusive)
        self.high_level_layers = mobilenet.features[4:]  # From layer 5 to the end

        self.project_low_level = nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        low_level_features = self.low_level_layers(x)
        low_level_features = self.project_low_level(low_level_features)
        high_level_features = self.high_level_layers(low_level_features)
        return {'low_level': low_level_features, 'high_level': high_level_features}

class DeepLabV3PlusHead(nn.Module):
    def __init__(self, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabV3PlusHead, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(24, 48, 1, bias=False),  # Projecting low-level features to 48 channels
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.aspp = ASPP(1280, aspp_dilate)  # ASPP expects 1280 channels from MobileNetV2

        self.decoder = nn.Sequential(
            nn.Conv2d(48, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, padding=1, bias=False),  # Adjust channels to match input
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, features):
        low_level_features = self.project(features['low_level'])
        high_level_features = self.aspp(features['high_level'])
        high_level_features = F.interpolate(high_level_features, size=low_level_features.shape[-2:], mode='bilinear', align_corners=False)
        low_level_features = self.decoder(low_level_features)
        x = torch.cat([low_level_features, high_level_features], dim=1)
        x = self.classifier(x)
        return x
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabV3Plus(_SimpleSegmentationModel):
    def __init__(self, num_classes_object, num_classes_material):
        backbone = MobileNetV2Backbone(pretrained=True)
        classifier_object = DeepLabV3PlusHead(num_classes_object)
        classifier_material = DeepLabV3PlusHead(num_classes_material)
        super(DeepLabV3Plus, self).__init__(backbone, classifier_object, classifier_material)

def get_model(num_classes_object, num_classes_material):
    return DeepLabV3Plus(num_classes_object, num_classes_material)

if __name__ == "__main__":
    num_classes_object = 21  # Example number of classes for PASCAL VOC 2012
    num_classes_material = 4  # Example number of material classes (set as needed)
    model = get_model(num_classes_object, num_classes_material)
    print(model)

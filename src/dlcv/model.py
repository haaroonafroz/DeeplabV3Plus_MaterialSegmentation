import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

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

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetBackbone, self).__init__()
        resnet101 = models.resnet101(weights='IMAGENET1K_V2' if pretrained else None)

        self.low_level_layers = nn.Sequential(
            resnet101.conv1,
            resnet101.bn1,
            resnet101.relu,
            resnet101.maxpool,
            resnet101.layer1  # Low-level features
        )
        self.high_level_layers = nn.Sequential(
            resnet101.layer2,
            resnet101.layer3,
            resnet101.layer4  # High-level features
        )

    def forward(self, x):
        low_level_features = self.low_level_layers(x)
        high_level_features = self.high_level_layers(low_level_features)
        return {'low_level': low_level_features, 'out': high_level_features}

class DeepLabV3PlusHead(nn.Module):
    def __init__(self, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabV3PlusHead, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.aspp = ASPP(2048, aspp_dilate)
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, features):
        low_level_features = self.project(features['low_level'])
        high_level_features = self.aspp(features['out'])
        high_level_features = F.interpolate(high_level_features, size=low_level_features.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier(torch.cat([low_level_features, high_level_features], dim=1))
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabV3Plus(_SimpleSegmentationModel):
    def __init__(self, num_classes):
        backbone = ResNetBackbone(pretrained=True)
        classifier = DeepLabV3PlusHead(num_classes)
        super(DeepLabV3Plus, self).__init__(backbone, classifier)

def get_model(num_classes):
    return DeepLabV3Plus(num_classes)

if __name__ == "__main__":
    num_classes = 21  # Example number of classes for PASCAL VOC 2012
    model = get_model(num_classes)
    print(model)

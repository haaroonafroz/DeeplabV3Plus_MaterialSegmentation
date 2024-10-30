import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context

# ============================DeeplabV3Plus=============================

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier_material):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier_material = classifier_material
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x_material = self.classifier_material(features)
        x_material = F.interpolate(x_material, size=input_shape, mode='bilinear', align_corners=False)
        return x_material

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

class ResNetBackbone(nn.Module):
    def __init__(self, resnet_type='resnet50', pretrained=True):
        super(ResNetBackbone, self).__init__()
        if resnet_type == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
        elif resnet_type == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported resnet_type: {resnet_type}")

        self.low_level_layers = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
        )
        self.high_level_layers = nn.Sequential(
            resnet.layer2, resnet.layer3, resnet.layer4
        )

    def forward(self, x):
        low_level_features = self.low_level_layers(x)
        high_level_features = self.high_level_layers(low_level_features)
        return {'low_level': low_level_features, 'high_level': high_level_features}

class DeepLabV3Plus(_SimpleSegmentationModel):
    def __init__(self, num_classes_material, backbone='mobilenet', freeze_backbone=True):
        if backbone == 'mobilenet':
            backbone_model = MobileNetV2Backbone(pretrained=True)
        elif backbone in ['resnet50', 'resnet101']:
            backbone_model = ResNetBackbone(resnet_type=backbone, pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        classifier_material = DeepLabV3PlusHead(num_classes_material)
        super(DeepLabV3Plus, self).__init__(backbone_model, classifier_material)
        
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self, freeze=True):
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
def get_model(num_classes_material, backbone='mobilenet', freeze_backbone=True):
    return DeepLabV3Plus(num_classes_material, backbone=backbone, freeze_backbone=freeze_backbone)

# ==========================U-Net==============================================================

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        # MobileNetV2 as encoder (pre-trained on ImageNet)
        mobilenet = models.mobilenet_v2(pretrained=True)

        # Use specific layers of MobileNetV2 as the encoder
        self.encoder1 = mobilenet.features[:4]   # Low-level features (stage 1)
        self.encoder2 = mobilenet.features[4:7]  # Mid-level features (stage 2)
        self.encoder3 = mobilenet.features[7:14] # High-level features (stage 3)
        self.encoder4 = mobilenet.features[14:]  # Deeper features (stage 4)

        # Decoder (upsampling layers with skip connections)
        self.upconv1 = nn.ConvTranspose2d(1280, 320, 2, stride=2)  # Stage 4 to Stage 3
        self.upconv2 = nn.ConvTranspose2d(416, 96, 2, stride=2)  # Stage 3 to Stage 2
        self.upconv3 = nn.ConvTranspose2d(128, 32, 2, stride=2)  # Stage 2 to Stage 1 (adjust input channels here)
        self.upconv4 = nn.ConvTranspose2d(56, 24, 2, stride=2)  # Stage 1 to final (adjust input channels here)

        # Final segmentation layer
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder forward pass with saved skip connections
        x1 = self.encoder1(x)  # Stage 1
        x2 = self.encoder2(x1) # Stage 2
        x3 = self.encoder3(x2) # Stage 3
        x4 = self.encoder4(x3) # Stage 4 (deepest features)

        # Decoder forward pass with skip connections
        x = self.upconv1(x4)                   # Stage 4 to Stage 3
        x = torch.cat([x, x3], dim=1)          # Concatenate skip connection
        
        x = self.upconv2(x)                    # Stage 3 to Stage 2
        x = torch.cat([x, x2], dim=1)          # Concatenate skip connection
        
        x = self.upconv3(x)                    # Stage 2 to Stage 1
        x = torch.cat([x, x1], dim=1)          # Concatenate skip connection
        
        x = self.upconv4(x)                    # Final upsampling

        # Final segmentation output
        x = self.final_conv(x)
        return x

def get_model_unet(num_classes):
    return UNet(num_classes)


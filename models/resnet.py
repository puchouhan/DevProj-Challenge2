import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def resnet14_audio(num_classes=50, input_channels=3):
    """Konstruiert ein ResNet-14 Modell für Audiodaten mit ResidualBlock mit verbesserter Regularisierung."""
    model = ResNet(ResidualBlock, [1, 1, 1, 1], num_classes=num_classes)

    # Anpassung des ersten Convolutional Layers für input_channels
    model.conv1 = nn.Sequential(
        nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Dropout2d(p=config.dropout_rate))

    # Füge Dropout zu den Ausgabeschichten hinzu
    model.fc = nn.Sequential(
        nn.Dropout(p=config.dropout_rate),
        nn.Linear(512, num_classes)
    )

    # Verbesserte Gewichtsinitialisierung
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    return model


def resnet18_audio(num_classes=50, input_channels=3):
    """Konstruiert ein ResNet-18 Modell für Audiodaten mit ResidualBlock."""
    model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=num_classes)
    # Anpassung des ersten Convolutional Layers für input_channels
    model.conv1 = nn.Sequential(
        nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU())
    return model

def resnet34_audio(num_classes=50, input_channels=3):
    """Konstruiert ein ResNet-34 Modell für Audiodaten mit ResidualBlock."""
    model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=num_classes)
    # Anpassung des ersten Convolutional Layers für input_channels
    model.conv1 = nn.Sequential(
        nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU())
    return model

def resnet50_audio(num_classes=50, input_channels=3):
    """Konstruiert ein ResNet-50 ähnliches Modell für Audiodaten."""
    model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=num_classes)
    # Anpassung des ersten Convolutional Layers für input_channels
    model.conv1 = nn.Sequential(
        nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU())
    return model

def resnet101_audio(num_classes=50, input_channels=3):
    """Konstruiert ein ResNet-101 ähnliches Modell für Audiodaten."""
    model = ResNet(ResidualBlock, [3, 4, 23, 3], num_classes=num_classes)
    # Anpassung des ersten Convolutional Layers für input_channels
    model.conv1 = nn.Sequential(
        nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU())
    return model

def resnet152_audio(num_classes=50, input_channels=3):
    """Konstruiert ein ResNet-152 ähnliches Modell für Audiodaten."""
    model = ResNet(ResidualBlock, [3, 8, 36, 3], num_classes=num_classes)
    # Anpassung des ersten Convolutional Layers für input_channels
    model.conv1 = nn.Sequential(
        nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU())
    return model
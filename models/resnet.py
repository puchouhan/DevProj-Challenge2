import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class BasicBlock(nn.Module):
    """
    Standard ResNet Basic Block mit Verbesserungen für Audio-Klassifikation.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Verbesserte Version des Conv-BN-ReLU-Blocks mit höherer Effizienz
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

        # Dropout für Regularisierung
        dropout_rate = getattr(config, 'dropout_rate', 0.1)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.dropout(out)  # Dropout nach der ersten Aktivierung für bessere Regularisierung
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet-Architektur optimiert für Audioklassifikation.
    """

    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64

        # Lese Eingangskanäle aus der Konfigurationsdatei
        in_channels = getattr(config, 'in_channels', 1)

        # Erste Konvolutionsschicht
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residualblöcke
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Globales Dropout vor der Klassifikationsschicht
        dropout_rate = getattr(config, 'dropout_rate', 0.2)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.fc = nn.Linear(512, num_classes)

        # Gewichtsinitialisierung für bessere Konvergenz
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Erstellt eine Schicht aus mehreren Residualblöcken.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        # Erster Block mit möglichem Downsampling
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes
        # Restliche Blöcke
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward-Pass durch das Netzwerk.
        """
        # Eingangsverarbeitung
        x = self.conv1(x)
        x = self.maxpool(x)

        # Residualblöcke
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Global Average Pooling und Klassifikation
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
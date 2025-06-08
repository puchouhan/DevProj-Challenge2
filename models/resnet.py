import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class BasicBlock(nn.Module):
    """
    Standard ResNet Basic Block mit Verbesserungen für Audio-Klassifikation.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=None, use_se=False):
        super(BasicBlock, self).__init__()
        # Verwende die Dropout-Rate aus der Konfiguration, falls nicht explizit angegeben
        self.dropout_rate = dropout_rate if dropout_rate is not None else getattr(config, 'dropout_rate', 0.1)

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
        self.dropout = nn.Dropout2d(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

        # Optional Squeeze-and-Excitation-Block für verbesserte Feature-Aufmerksamkeit
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(out_channels, reduction=16)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)

        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation Block für kanalweise Aufmerksamkeit.
    """

    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNet(nn.Module):
    """
    ResNet-Architektur optimiert für Audioklassifikation.

    Unterstützt:
    - Dynamische Kanalanzahl-Anpassung
    - Optionale Dropout-Regularisierung
    - Batch-Normalisierung
    - Squeeze-and-Excitation für verbesserte Feature-Aufmerksamkeit
    - Optimierte Merkmalslernfähigkeit durch Architekturanpassungen
    """

    def __init__(self, block, layers, num_classes=10, in_channels=None, dropout_rate=None,
                 use_se=False, initial_filters=64, zero_init_residual=False):
        """
        Initialisiert das ResNet-Modell.

        Args:
            block: Blocktyp (BasicBlock oder Bottleneck)
            layers: Liste mit Anzahl der Blöcke pro Schicht
            num_classes: Anzahl der Ausgabeklassen
            in_channels: Anzahl der Eingangskanäle (1 für Mel-Spektrogramme)
            dropout_rate: Dropout-Rate für Regularisierung
            use_se: Ob Squeeze-and-Excitation-Blöcke verwendet werden sollen
            initial_filters: Anzahl der Filter in der ersten Konvolutionsschicht
            zero_init_residual: Ob Residualverbindungen mit Null initialisiert werden sollen
        """
        super(ResNet, self).__init__()

        # Lese Parameter aus der Config-Datei, wenn nicht explizit angegeben
        self.in_channels = in_channels if in_channels is not None else getattr(config, 'in_channels', 1)
        self.dropout_rate = dropout_rate if dropout_rate is not None else getattr(config, 'dropout_rate', 0.2)

        self.inplanes = initial_filters

        # Erste Konvolutionsschicht angepasst an die Eingangskanäle
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, initial_filters, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(initial_filters),
            nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residualblöcke
        self.layer0 = self._make_layer(block, initial_filters, layers[0], stride=1,
                                       dropout_rate=self.dropout_rate, use_se=use_se)
        self.layer1 = self._make_layer(block, initial_filters * 2, layers[1], stride=2,
                                       dropout_rate=self.dropout_rate, use_se=use_se)
        self.layer2 = self._make_layer(block, initial_filters * 4, layers[2], stride=2,
                                       dropout_rate=self.dropout_rate, use_se=use_se)
        self.layer3 = self._make_layer(block, initial_filters * 8, layers[3], stride=2,
                                       dropout_rate=self.dropout_rate, use_se=use_se)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        self.fc = nn.Linear(initial_filters * 8, num_classes)

        # Gewichtsinitialisierung
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Optionale Initialisierung der Residualblöcke mit Null
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.conv2[1].weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dropout_rate=0.0, use_se=False):
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
        layers.append(block(self.inplanes, planes, stride, downsample,
                            dropout_rate=dropout_rate, use_se=use_se))

        self.inplanes = planes
        # Restliche Blöcke
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_rate=dropout_rate, use_se=use_se))

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
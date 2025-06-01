import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, dropout_rate=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate else nn.Identity()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, dropout_rate=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate else nn.Identity()
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetAudio(nn.Module):
    def __init__(self, block, layers, num_classes=50, input_channels=1, dropout_rate=None):
        super(ResNetAudio, self).__init__()
        self.in_planes = 64
        self.dropout_rate = dropout_rate

        # Input layer: Conv2d for single-channel spectrogram
        # Spectrograms are (batch, 1, n_mels, n_steps)
        # n_mels is height, n_steps is width
        self.conv1 = nn.Conv2d(input_channels, self.in_planes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout_rate=dropout_rate)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1, dropout_rate=None):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers_list = []
        layers_list.append(block(self.in_planes, planes, stride, downsample, dropout_rate))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers_list.append(block(self.in_planes, planes, dropout_rate=dropout_rate))

        return nn.Sequential(*layers_list)

    def forward(self, x):
        # x is expected to be (batch_size, 1, n_mels, n_steps)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


# Helper functions to create specific ResNetAudio models
def resnet18_audio(num_classes=50, input_channels=1):
    """Constructs a ResNet-18 model for audio."""
    import config
    dropout_rate = config.dropout_rate if hasattr(config, 'dropout_rate') else None
    return ResNetAudio(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, 
                      input_channels=input_channels, dropout_rate=dropout_rate)


def resnet34_audio(num_classes=50, input_channels=1):
    """Constructs a ResNet-34 model for audio."""
    import config
    dropout_rate = config.dropout_rate if hasattr(config, 'dropout_rate') else None
    return ResNetAudio(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, 
                      input_channels=input_channels, dropout_rate=dropout_rate)

def resnet14_audio(num_classes=50, input_channels=1):
    """Constructs a ResNet-14 model for audio."""
    import config
    dropout_rate = config.dropout_rate if hasattr(config, 'dropout_rate') else None
    return ResNetAudio(BasicBlock, [2, 2, 1, 1], num_classes=num_classes, 
                      input_channels=input_channels, dropout_rate=dropout_rate)

def resnet50_audio(num_classes=50, input_channels=1):
    """Constructs a ResNet-50 model for audio."""
    import config
    dropout_rate = config.dropout_rate if hasattr(config, 'dropout_rate') else None
    return ResNetAudio(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, 
                      input_channels=input_channels, dropout_rate=dropout_rate)

def resnet101_audio(num_classes=50, input_channels=1):
    """Constructs a ResNet-101 model for audio."""
    import config
    dropout_rate = config.dropout_rate if hasattr(config, 'dropout_rate') else None
    return ResNetAudio(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, 
                      input_channels=input_channels, dropout_rate=dropout_rate)

def resnet152_audio(num_classes=50, input_channels=1):
    """Constructs a ResNet-152 model for audio."""
    import config
    dropout_rate = config.dropout_rate if hasattr(config, 'dropout_rate') else None
    return ResNetAudio(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, 
                      input_channels=input_channels, dropout_rate=dropout_rate)

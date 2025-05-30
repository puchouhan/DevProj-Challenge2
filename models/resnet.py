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

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetAudio(nn.Module):
    def __init__(self, block, layers, num_classes=50, input_channels=1):
        super(ResNetAudio, self).__init__()
        self.in_planes = 64

        # Input layer: Conv2d for single-channel spectrogram
        # Spectrograms are (batch, 1, n_mels, n_steps)
        # n_mels is height, n_steps is width
        self.conv1 = nn.Conv2d(input_channels, self.in_planes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers_list = []
        layers_list.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers_list.append(block(self.in_planes, planes))

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
        x = self.fc(x)

        return x


# Helper functions to create specific ResNetAudio models (e.g., ResNet18-like)
def resnet18_audio(num_classes=50, input_channels=1):
    """Constructs a ResNet-18 model for audio."""
    return ResNetAudio(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, input_channels=input_channels)


def resnet34_audio(num_classes=50, input_channels=1):
    """Constructs a ResNet-34 model for audio."""
    return ResNetAudio(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, input_channels=input_channels)

# Example of how you might define it in your config.py:
# model_constructor = "resnet.resnet18_audio(num_classes=config.n_classes)"
#
# Or, if you want to pass n_mels and n_steps (though not strictly needed with AdaptiveAvgPool2d):
# class ResNetAudioCustom(ResNetAudio):
#     def __init__(self, n_steps, n_mels, output_size, block_type=BasicBlock, layers=[2,2,2,2], **kwargs):
#         # n_steps and n_mels are not directly used by the ResNetAudio base if AdaptiveAvgPool2d is used
#         # but can be kept for interface consistency if desired.
#         super().__init__(block=block_type, layers=layers, num_classes=output_size, input_channels=1)

# Note: The existing ResNet class in the initial problem description was an MLP.
# This implementation provides a proper ResNet CNN structure.
# You would replace the content of your resnet.py with this code.
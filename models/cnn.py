
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Convolutional block mit BatchNorm und optional Dropout"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_rate=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class AudioCNN(nn.Module):
    def __init__(self, num_classes=50, input_channels=1, dropout_rate=None):
        super(AudioCNN, self).__init__()
        
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()
        
        # Erste Konvolution mit größerem Kernel für Übersicht
        self.layer1 = ConvBlock(input_channels, 64, kernel_size=7, stride=2, padding=3, dropout_rate=dropout_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Convolutional Blöcke mit steigender Feature-Tiefe
        self.layer2 = nn.Sequential(
            ConvBlock(64, 64, dropout_rate=dropout_rate),
            ConvBlock(64, 64, dropout_rate=dropout_rate),
            ConvBlock(64, 128, stride=2, dropout_rate=dropout_rate),
        )
        
        self.layer3 = nn.Sequential(
            ConvBlock(128, 128, dropout_rate=dropout_rate),
            ConvBlock(128, 128, dropout_rate=dropout_rate),
            ConvBlock(128, 256, stride=2, dropout_rate=dropout_rate),
        )
        
        self.layer4 = nn.Sequential(
            ConvBlock(256, 256, dropout_rate=dropout_rate),
            ConvBlock(256, 256, dropout_rate=dropout_rate),
            ConvBlock(256, 512, stride=2, dropout_rate=dropout_rate),
        )
        
        # Global Pooling und Klassifikations-Layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Gewichtsinitalisierung
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Input: [batch_size, 1, n_mels, n_steps]
        
        x = self.layer1(x)
        x = self.pool1(x)
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# Verschiedene CNN-Varianten mit unterschiedlicher Komplexität

def cnn_small_audio(num_classes=50, input_channels=1):
    """Kleine CNN-Variante für Audio"""
    import config
    dropout_rate = config.dropout_rate if hasattr(config, 'dropout_rate') else None
    model = AudioCNN(num_classes=num_classes, input_channels=input_channels, dropout_rate=dropout_rate)
    return model

def cnn_medium_audio(num_classes=50, input_channels=1):
    """Mittlere CNN-Variante für Audio mit mehr Faltungsschichten"""
    import config
    dropout_rate = config.dropout_rate if hasattr(config, 'dropout_rate') else None
    model = AudioCNN(num_classes=num_classes, input_channels=input_channels, dropout_rate=dropout_rate)
    # Hier könntest du die Architektur anpassen, falls nötig
    return model

def cnn_large_audio(num_classes=50, input_channels=1):
    """Große CNN-Variante für Audio mit tiefer Architektur"""
    import config
    dropout_rate = config.dropout_rate if hasattr(config, 'dropout_rate') else None
    model = AudioCNN(num_classes=num_classes, input_channels=input_channels, dropout_rate=dropout_rate)
    # Hier könntest du die Architektur anpassen, falls nötig
    return model

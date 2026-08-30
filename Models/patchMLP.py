import torch
import torch.nn as nn


class PatchMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_layers=None, activation="ReLU", dropout=0.0):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [32, 16]
        self.input_dim = int(input_dim)
        self.output_size = int(num_classes)
        self.hidden_size = [int(i * self.input_dim) for i in hidden_layers]
        activation = str(activation).lower()
        if activation == "LeakyReLu":
            self.act_layer = nn.LeakyReLU()
        elif activation == "tanh":
            self.act_layer = nn.Tanh()
        elif activation == "gelu":
            self.act_layer = nn.GELU()
        else:
            self.act_layer = nn.ReLU()


        self.input = nn.Linear(self.input_dim, self.hidden_size[0])

        self.hidden = nn.ModuleList()

        for i in range(len(hidden_layers) - 1):
            self.hidden.append(nn.Linear(self.hidden_size[i], self.hidden_size[i+1]))

        self.output = nn.Linear(self.hidden_size[-1], self.output_size)    


    def forward(self, x):
        x = self.input(x)
        x = self.act_layer(x)
        for i in range(len(self.hidden)):
            x = self.hidden[i](x)
            x = self.act_layer(x)
        x = self.output(x)

        return x


class WVSpecPatchCNN(nn.Module):
    """
    Spectral patch-level CNN classifier for multispectral split-image patches.
    Input: (B, 8, H, W)
    Output: (B, num_classes)
    """
    def __init__(self, in_channels=8, num_classes=4, base_filters=32, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(in_channels, base_filters),
            nn.MaxPool2d(2),
            self._conv_block(base_filters, base_filters * 2),
            nn.MaxPool2d(2),
            self._conv_block(base_filters * 2, base_filters * 4),
            nn.MaxPool2d(2),
            self._conv_block(base_filters * 4, base_filters * 4),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_filters * 4, num_classes)
        )

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class WVSplitUNet(nn.Module):
    """
    Encoder-decoder CNN for dense spatial feature detection.
    Input: (B, 8, 300, 400)
    Output: (B, num_classes, 300, 400) — per-pixel class predictions
    """
    def __init__(self, in_channels=8, num_classes=4):
        super().__init__()
        
        # Encoder (downsampling with skip connections)
        self.enc1 = self._conv_block(in_channels, 32)          # 8 -> 32
        self.pool1 = nn.MaxPool2d(2, stride=2)                 # 300x400 -> 150x200
        
        self.enc2 = self._conv_block(32, 64)                   # 32 -> 64
        self.pool2 = nn.MaxPool2d(2, stride=2)                 # 150x200 -> 75x100
        
        self.enc3 = self._conv_block(64, 128)                  # 64 -> 128
        self.pool3 = nn.MaxPool2d(2, stride=2)                 # 75x100 -> 38x50
        
        self.bottleneck = self._conv_block(128, 256)           # 128 -> 256
        
        # Decoder (upsampling with skip connections)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(256, 128)  # concatenate skip from enc3
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)   # concatenate skip from enc2
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64, 32)    # concatenate skip from enc1
        
        # Final output: 1x1 conv to num_classes
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        # Use output_size to handle odd dimensions from pooling
        x = self.upconv3(x, output_size=(enc3.shape[2], enc3.shape[3]))
        x = torch.cat([x, enc3], dim=1)  # concatenate skip
        x = self.dec3(x)
        
        x = self.upconv2(x, output_size=(enc2.shape[2], enc2.shape[3]))
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x, output_size=(enc1.shape[2], enc1.shape[3]))
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        x = self.final(x)
        return x
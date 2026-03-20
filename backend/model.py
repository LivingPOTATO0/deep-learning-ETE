import torch
import torch.nn as nn

def conv_block(in_ch, out_ch, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=kernel, stride=1, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

class DamageNet(nn.Module):
    def __init__(self, num_classes=3, dropout_p=0.5):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),    # -> 32 x 112 x 112
            conv_block(32, 64),   # -> 64 x 56 x 56
            conv_block(64, 128),  # -> 128 x 28 x 28
            conv_block(128, 256), # -> 256 x 14 x 14
            conv_block(256, 512), # -> 512 x 7 x 7
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p*0.5),
            nn.Linear(256, num_classes)
        )
        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
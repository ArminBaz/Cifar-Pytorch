import torch
import torch.nn as nn

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        # Convolutional Blocks
        self.conv_blocks = nn.Sequential(
                # Block 1
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding = 1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                # Block 2
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(p=0.05),
            
                # Block 3
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully Connected Block
        self.fc_block = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    # forward pass function
    def forward(self, x):
        # conv layers
        x = self.conv_blocks(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_block(x)

        return x

import torch
import torch.nn as nn

class VGG(nn.Module):

    def __init__(self, input_channel, num_class):
        super().__init__()
        self.conv = nn.Sequential(
            # 32 32 3 (입력)
            # 32 32 64
            nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16 16 64

            # 16 16 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8 8 128

            # 8 8 256
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4 4 256

            # 4 4 512
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=512), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=512), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 2 2 512
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=512*2*2, out_features=2048), nn.ReLU(inplace=True),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=2048, out_features=num_class)
        )

        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size()[0], -1) # 256, 512*2*2
        output = self.fc(conv_out)
        return output
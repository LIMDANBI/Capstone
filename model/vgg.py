import torch
import torch.nn as nn

class VGG16(nn.Module):

    def __init__(self, input_channel, num_class):
        super().__init__()
        self.conv = nn.Sequential(
            # 32 32 3 (입력)
            # 32 32 64
            nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=64 , out_channels=64 , kernel_size=3 , stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 16 16 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 8 8 256
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 4 4 512
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # 3 3 512
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1) # 2 2 512
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=512*2*2, out_features=4096), nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096), nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_class)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size()[0], -1) # 256, 512*2*2
        fc_out = self.fc(conv_out)
        output = self.softmax(fc_out)

        return output
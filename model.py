#code 2
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()

        # C1:
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1),  
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=1), 
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=1), 
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=20, out_channels=32, kernel_size=(3,3),stride=2,dilation=2,padding=1),  #Downsampling
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

        )

        # C2:
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=48, kernel_size=(3, 3), dilation=2, stride =2, padding=2),  # Dilated convolution, Downsampling
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(0.1),


        )

        # C3:
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3, 3), dilation=2, padding=2,stride=2),  # Dilated convolution, Downsampling
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),

        )

        # C4:
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1,dilation=2), #added dilation
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=1,groups=32), #Depthwise convolution
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1)),  # Pointwise convolution
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.05),

        )

        # OUTPUT: Global Average Pooling
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1))  # Adjust kernel size based on input dimensions
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor to [batch_size, 10]
        return F.log_softmax(x, dim=1)

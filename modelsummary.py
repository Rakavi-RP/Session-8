pip install torchsummary
from torchsummary import summary

model = CIFAR10()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
summary(model, input_size=(3, 32, 32))


Requirement already satisfied: torchsummary in /usr/local/lib/python3.10/dist-packages (1.5.1)
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 32, 32]             224
       BatchNorm2d-2            [-1, 8, 32, 32]              16
              ReLU-3            [-1, 8, 32, 32]               0
           Dropout-4            [-1, 8, 32, 32]               0
            Conv2d-5           [-1, 10, 32, 32]             730
       BatchNorm2d-6           [-1, 10, 32, 32]              20
              ReLU-7           [-1, 10, 32, 32]               0
           Dropout-8           [-1, 10, 32, 32]               0
            Conv2d-9           [-1, 20, 32, 32]           1,820
      BatchNorm2d-10           [-1, 20, 32, 32]              40
             ReLU-11           [-1, 20, 32, 32]               0
          Dropout-12           [-1, 20, 32, 32]               0
           Conv2d-13           [-1, 32, 15, 15]           5,792
      BatchNorm2d-14           [-1, 32, 15, 15]              64
             ReLU-15           [-1, 32, 15, 15]               0
          Dropout-16           [-1, 32, 15, 15]               0
           Conv2d-17           [-1, 48, 15, 15]          13,872
      BatchNorm2d-18           [-1, 48, 15, 15]              96
             ReLU-19           [-1, 48, 15, 15]               0
          Dropout-20           [-1, 48, 15, 15]               0
           Conv2d-21           [-1, 64, 15, 15]          27,712
      BatchNorm2d-22           [-1, 64, 15, 15]             128
             ReLU-23           [-1, 64, 15, 15]               0
          Dropout-24           [-1, 64, 15, 15]               0
           Conv2d-25             [-1, 48, 8, 8]          27,696
      BatchNorm2d-26             [-1, 48, 8, 8]              96
             ReLU-27             [-1, 48, 8, 8]               0
          Dropout-28             [-1, 48, 8, 8]               0
           Conv2d-29             [-1, 32, 8, 8]          13,856
      BatchNorm2d-30             [-1, 32, 8, 8]              64
             ReLU-31             [-1, 32, 8, 8]               0
          Dropout-32             [-1, 32, 8, 8]               0
           Conv2d-33             [-1, 64, 8, 8]          18,496
      BatchNorm2d-34             [-1, 64, 8, 8]             128
             ReLU-35             [-1, 64, 8, 8]               0
          Dropout-36             [-1, 64, 8, 8]               0
           Conv2d-37             [-1, 16, 4, 4]           9,232
      BatchNorm2d-38             [-1, 16, 4, 4]              32
             ReLU-39             [-1, 16, 4, 4]               0
          Dropout-40             [-1, 16, 4, 4]               0
           Conv2d-41             [-1, 32, 4, 4]           4,640
      BatchNorm2d-42             [-1, 32, 4, 4]              64
             ReLU-43             [-1, 32, 4, 4]               0
          Dropout-44             [-1, 32, 4, 4]               0
           Conv2d-45             [-1, 32, 2, 2]           9,248
      BatchNorm2d-46             [-1, 32, 2, 2]              64
             ReLU-47             [-1, 32, 2, 2]               0
          Dropout-48             [-1, 32, 2, 2]               0
           Conv2d-49             [-1, 64, 2, 2]             640
           Conv2d-50             [-1, 64, 2, 2]           4,160
      BatchNorm2d-51             [-1, 64, 2, 2]             128
             ReLU-52             [-1, 64, 2, 2]               0
          Dropout-53             [-1, 64, 2, 2]               0
AdaptiveAvgPool2d-54             [-1, 64, 1, 1]               0
================================================================
Total params: 139,058
Trainable params: 139,058
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 2.50
Params size (MB): 0.53
Estimated Total Size (MB): 3.04
----------------------------------------------------------------

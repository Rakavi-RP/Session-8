# CIFAR-10 Classification with PyTorch

## Introduction

This project implements a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset using PyTorch. It includes modularized scripts for data loading, model architecture, training, and evaluation.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. Classes include:Airplane,Automobile,Bird,Cat,Deer,Dog,Frog,Horse,Ship,Truck.

## Requirements
 Python 3.7+
 
 PyTorch 1.10+
 
 Albumentations 1.3.0
 
 TorchSummary
 
 tqdm

## Project Structure

```bash

├── Session8.ipynb       # Entire Colab code with logs
├── dataloader.py        # Handles data loading and augmentations
├── model.py             # Defines the CNN architecture
├── train.py             # Defines functions for model training and testing
├── main.py              # Runs model testing and training

```

## Model Architecture

The CNN architecture consists of four convolutional blocks, each followed by Batch Normalization, ReLU activation, and Dropout. Depthwise separable and dilated convolutions are also utilized. Global Average Pooling (GAP) is applied before the final output layer.

## Block Details
```bash
| Block           | Operation Details                                 | Input Size    | Output Size   |
|-----------------|---------------------------------------------------|---------------|---------------|
| **Input Layer** | -                                                 | (3, 32, 32)   | (3, 32, 32)   |
| **C1 Block**    | Conv2d -> BatchNorm2d -> ReLU -> Dropout          | (3, 32, 32)   | (32, 16, 16)  |
| **C2 Block**    | Conv2d -> BatchNorm2d -> ReLU -> Dropout          | (32, 16, 16)  | (48, 8, 8)    |
| **C3 Block**    | Conv2d -> BatchNorm2d -> ReLU -> Dropout          | (48, 8, 8)    | (16, 4, 4)    |
| **C4 Block**    | Depthwise Separable Conv -> BatchNorm2d -> ReLU   | (16, 4, 4)    | (64, 4, 4)    |
| **GAP**         | AdaptiveAvgPool2d((1, 1))                         | (64, 4, 4)    | (64, 1, 1)    |
| **Flatten**     | Tensor reshaped to [batch_size, 64]               | (64, 1, 1)    | (64)          |
| **Output Layer**| Log-Softmax Activation                            | (64)          | (10)          |

```

## Model Summary
```bash
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
```

## Hyperparameters

Optimizer: Adam (Learning Rate: 0.005, Weight Decay: 1e-4)

Scheduler: ReduceLROnPlateau (Factor: 0.75, Patience: 3)

Batch Size: 128 (or 64 if CUDA unavailable)

Epochs: 150

## Logs

```bash
EPOCH: 1
Loss=1.784204125404358 Batch_id=390 Accuracy=30.38: 100%|██████████| 391/391 [00:22<00:00, 17.16it/s]

Test set: Average loss: 1.8098, Accuracy: 4465/10000 (44.65%)

EPOCH: 2
Loss=1.9006723165512085 Batch_id=390 Accuracy=38.89: 100%|██████████| 391/391 [00:20<00:00, 19.02it/s]

Test set: Average loss: 1.7824, Accuracy: 4194/10000 (41.94%)

EPOCH: 3
Loss=1.606075644493103 Batch_id=390 Accuracy=42.69: 100%|██████████| 391/391 [00:21<00:00, 18.57it/s]

Test set: Average loss: 1.4072, Accuracy: 5126/10000 (51.26%)

EPOCH: 4
Loss=1.5071799755096436 Batch_id=390 Accuracy=45.11: 100%|██████████| 391/391 [00:21<00:00, 17.90it/s]

Test set: Average loss: 1.2706, Accuracy: 5616/10000 (56.16%)

EPOCH: 5
Loss=1.6294333934783936 Batch_id=390 Accuracy=46.85: 100%|██████████| 391/391 [00:20<00:00, 18.67it/s]

Test set: Average loss: 1.1787, Accuracy: 5989/10000 (59.89%)

EPOCH: 6
Loss=1.5707405805587769 Batch_id=390 Accuracy=48.68: 100%|██████████| 391/391 [00:20<00:00, 18.85it/s]

Test set: Average loss: 1.2559, Accuracy: 5770/10000 (57.70%)

EPOCH: 7
Loss=1.7671916484832764 Batch_id=390 Accuracy=50.36: 100%|██████████| 391/391 [00:22<00:00, 17.08it/s]

Test set: Average loss: 1.0575, Accuracy: 6400/10000 (64.00%)

EPOCH: 8
Loss=1.2973809242248535 Batch_id=390 Accuracy=51.41: 100%|██████████| 391/391 [00:21<00:00, 18.45it/s]

Test set: Average loss: 1.0127, Accuracy: 6637/10000 (66.37%)

EPOCH: 9
Loss=1.4158809185028076 Batch_id=390 Accuracy=52.42: 100%|██████████| 391/391 [00:21<00:00, 18.48it/s]

Test set: Average loss: 0.9738, Accuracy: 6722/10000 (67.22%)

EPOCH: 10
Loss=1.1812047958374023 Batch_id=390 Accuracy=53.68: 100%|██████████| 391/391 [00:21<00:00, 18.59it/s]

Test set: Average loss: 0.9258, Accuracy: 6866/10000 (68.66%)

EPOCH: 11
Loss=1.3768517971038818 Batch_id=390 Accuracy=54.39: 100%|██████████| 391/391 [00:20<00:00, 19.37it/s]

Test set: Average loss: 0.9313, Accuracy: 6852/10000 (68.52%)

EPOCH: 12
Loss=1.412545084953308 Batch_id=390 Accuracy=55.33: 100%|██████████| 391/391 [00:20<00:00, 19.17it/s]

Test set: Average loss: 0.9776, Accuracy: 6749/10000 (67.49%)

EPOCH: 13
Loss=1.263278603553772 Batch_id=390 Accuracy=55.76: 100%|██████████| 391/391 [00:21<00:00, 18.07it/s]

Test set: Average loss: 0.9459, Accuracy: 6928/10000 (69.28%)

EPOCH: 14
Loss=1.2716405391693115 Batch_id=390 Accuracy=56.43: 100%|██████████| 391/391 [00:20<00:00, 18.86it/s]

Test set: Average loss: 0.8467, Accuracy: 7145/10000 (71.45%)

EPOCH: 15
Loss=1.4615366458892822 Batch_id=390 Accuracy=56.23: 100%|██████████| 391/391 [00:23<00:00, 16.48it/s]

Test set: Average loss: 0.9183, Accuracy: 6917/10000 (69.17%)

EPOCH: 16
Loss=1.4782698154449463 Batch_id=390 Accuracy=57.20: 100%|██████████| 391/391 [00:21<00:00, 18.47it/s]

Test set: Average loss: 0.8801, Accuracy: 7010/10000 (70.10%)

EPOCH: 17
Loss=1.2419559955596924 Batch_id=390 Accuracy=57.31: 100%|██████████| 391/391 [00:21<00:00, 18.54it/s]

Test set: Average loss: 0.8591, Accuracy: 7131/10000 (71.31%)

EPOCH: 18
Loss=1.2528436183929443 Batch_id=390 Accuracy=57.39: 100%|██████████| 391/391 [00:21<00:00, 18.50it/s]

Test set: Average loss: 0.8815, Accuracy: 7045/10000 (70.45%)

EPOCH: 19
Loss=1.1059255599975586 Batch_id=390 Accuracy=58.74: 100%|██████████| 391/391 [00:20<00:00, 18.66it/s]

Test set: Average loss: 0.7836, Accuracy: 7400/10000 (74.00%)

EPOCH: 20
Loss=1.4265553951263428 Batch_id=390 Accuracy=59.35: 100%|██████████| 391/391 [00:22<00:00, 17.60it/s]

Test set: Average loss: 0.7733, Accuracy: 7460/10000 (74.60%)

EPOCH: 21
Loss=1.4967067241668701 Batch_id=390 Accuracy=59.98: 100%|██████████| 391/391 [00:21<00:00, 18.37it/s]

Test set: Average loss: 0.7770, Accuracy: 7395/10000 (73.95%)

EPOCH: 22
Loss=1.425014615058899 Batch_id=390 Accuracy=59.83: 100%|██████████| 391/391 [00:21<00:00, 18.47it/s]

Test set: Average loss: 0.7542, Accuracy: 7463/10000 (74.63%)

EPOCH: 23
Loss=1.3309645652770996 Batch_id=390 Accuracy=59.76: 100%|██████████| 391/391 [00:21<00:00, 17.93it/s]

Test set: Average loss: 0.7773, Accuracy: 7420/10000 (74.20%)

EPOCH: 24
Loss=1.3264484405517578 Batch_id=390 Accuracy=60.62: 100%|██████████| 391/391 [00:21<00:00, 18.55it/s]

Test set: Average loss: 0.7333, Accuracy: 7619/10000 (76.19%)

EPOCH: 25
Loss=0.9846401214599609 Batch_id=390 Accuracy=60.72: 100%|██████████| 391/391 [00:20<00:00, 19.14it/s]

Test set: Average loss: 0.7661, Accuracy: 7409/10000 (74.09%)

EPOCH: 26
Loss=1.1011568307876587 Batch_id=390 Accuracy=60.96: 100%|██████████| 391/391 [00:20<00:00, 19.21it/s]

Test set: Average loss: 0.7746, Accuracy: 7446/10000 (74.46%)

EPOCH: 27
Loss=0.9015341997146606 Batch_id=390 Accuracy=60.83: 100%|██████████| 391/391 [00:20<00:00, 19.30it/s]

Test set: Average loss: 0.7277, Accuracy: 7599/10000 (75.99%)

EPOCH: 28
Loss=1.4333510398864746 Batch_id=390 Accuracy=61.01: 100%|██████████| 391/391 [00:21<00:00, 18.47it/s]

Test set: Average loss: 0.7506, Accuracy: 7562/10000 (75.62%)

EPOCH: 29
Loss=1.31815505027771 Batch_id=390 Accuracy=61.22: 100%|██████████| 391/391 [00:20<00:00, 18.64it/s]

Test set: Average loss: 0.7361, Accuracy: 7575/10000 (75.75%)

EPOCH: 30
Loss=1.184014916419983 Batch_id=390 Accuracy=61.42: 100%|██████████| 391/391 [00:22<00:00, 17.31it/s]

Test set: Average loss: 0.7377, Accuracy: 7558/10000 (75.58%)

EPOCH: 31
Loss=1.0744478702545166 Batch_id=390 Accuracy=61.45: 100%|██████████| 391/391 [00:20<00:00, 18.62it/s]

Test set: Average loss: 0.8785, Accuracy: 7176/10000 (71.76%)

EPOCH: 32
Loss=1.0232740640640259 Batch_id=390 Accuracy=62.73: 100%|██████████| 391/391 [00:21<00:00, 18.57it/s]

Test set: Average loss: 0.6857, Accuracy: 7707/10000 (77.07%)

EPOCH: 33
Loss=1.269256830215454 Batch_id=390 Accuracy=63.13: 100%|██████████| 391/391 [00:21<00:00, 18.30it/s]

Test set: Average loss: 0.6432, Accuracy: 7876/10000 (78.76%)

EPOCH: 34
Loss=0.9485543966293335 Batch_id=390 Accuracy=63.46: 100%|██████████| 391/391 [00:20<00:00, 18.84it/s]

Test set: Average loss: 0.6649, Accuracy: 7798/10000 (77.98%)

EPOCH: 35
Loss=1.1523869037628174 Batch_id=390 Accuracy=63.46: 100%|██████████| 391/391 [00:20<00:00, 19.19it/s]

Test set: Average loss: 0.6628, Accuracy: 7804/10000 (78.04%)

EPOCH: 36
Loss=1.1720452308654785 Batch_id=390 Accuracy=63.17: 100%|██████████| 391/391 [00:21<00:00, 18.00it/s]

Test set: Average loss: 0.6926, Accuracy: 7731/10000 (77.31%)

EPOCH: 37
Loss=1.0660736560821533 Batch_id=390 Accuracy=63.30: 100%|██████████| 391/391 [00:21<00:00, 18.60it/s]

Test set: Average loss: 0.6842, Accuracy: 7755/10000 (77.55%)

EPOCH: 38
Loss=1.0470197200775146 Batch_id=390 Accuracy=64.21: 100%|██████████| 391/391 [00:24<00:00, 16.22it/s]

Test set: Average loss: 0.6398, Accuracy: 7894/10000 (78.94%)

EPOCH: 39
Loss=0.8215585947036743 Batch_id=390 Accuracy=64.85: 100%|██████████| 391/391 [00:20<00:00, 18.63it/s]

Test set: Average loss: 0.6347, Accuracy: 7894/10000 (78.94%)

EPOCH: 40
Loss=1.1809742450714111 Batch_id=390 Accuracy=65.14: 100%|██████████| 391/391 [00:21<00:00, 18.11it/s]

Test set: Average loss: 0.6242, Accuracy: 7933/10000 (79.33%)

EPOCH: 41
Loss=1.088869333267212 Batch_id=390 Accuracy=64.81: 100%|██████████| 391/391 [00:20<00:00, 18.79it/s]

Test set: Average loss: 0.6184, Accuracy: 7928/10000 (79.28%)

EPOCH: 42
Loss=1.0757888555526733 Batch_id=390 Accuracy=65.15: 100%|██████████| 391/391 [00:20<00:00, 18.90it/s]

Test set: Average loss: 0.5976, Accuracy: 8005/10000 (80.05%)

EPOCH: 43
Loss=1.066718339920044 Batch_id=390 Accuracy=65.16: 100%|██████████| 391/391 [00:21<00:00, 18.59it/s]

Test set: Average loss: 0.6590, Accuracy: 7823/10000 (78.23%)

EPOCH: 44
Loss=1.2584915161132812 Batch_id=390 Accuracy=65.16: 100%|██████████| 391/391 [00:22<00:00, 17.72it/s]

Test set: Average loss: 0.6112, Accuracy: 7989/10000 (79.89%)

EPOCH: 45
Loss=1.1435668468475342 Batch_id=390 Accuracy=65.22: 100%|██████████| 391/391 [00:23<00:00, 16.53it/s]

Test set: Average loss: 0.6109, Accuracy: 7981/10000 (79.81%)

EPOCH: 46
Loss=1.0901515483856201 Batch_id=390 Accuracy=65.19: 100%|██████████| 391/391 [00:22<00:00, 17.70it/s]

Test set: Average loss: 0.5979, Accuracy: 8027/10000 (80.27%)

EPOCH: 47
Loss=0.783997118473053 Batch_id=390 Accuracy=66.08: 100%|██████████| 391/391 [00:21<00:00, 18.48it/s]

Test set: Average loss: 0.5658, Accuracy: 8145/10000 (81.45%)

EPOCH: 48
Loss=1.1318625211715698 Batch_id=390 Accuracy=66.19: 100%|██████████| 391/391 [00:21<00:00, 18.46it/s]

Test set: Average loss: 0.6536, Accuracy: 7899/10000 (78.99%)

EPOCH: 49
Loss=1.2843406200408936 Batch_id=390 Accuracy=66.19: 100%|██████████| 391/391 [00:22<00:00, 17.63it/s]

Test set: Average loss: 0.5930, Accuracy: 8029/10000 (80.29%)

EPOCH: 50
Loss=0.9373900294303894 Batch_id=390 Accuracy=66.28: 100%|██████████| 391/391 [00:21<00:00, 18.21it/s]

Test set: Average loss: 0.5894, Accuracy: 8087/10000 (80.87%)

EPOCH: 51
Loss=0.9732037782669067 Batch_id=390 Accuracy=66.17: 100%|██████████| 391/391 [00:21<00:00, 17.93it/s]

Test set: Average loss: 0.5931, Accuracy: 8032/10000 (80.32%)

EPOCH: 52
Loss=1.1602511405944824 Batch_id=390 Accuracy=67.18: 100%|██████████| 391/391 [00:21<00:00, 18.50it/s]

Test set: Average loss: 0.5490, Accuracy: 8191/10000 (81.91%)

EPOCH: 53
Loss=1.120539903640747 Batch_id=390 Accuracy=67.18: 100%|██████████| 391/391 [00:25<00:00, 15.54it/s]

Test set: Average loss: 0.5629, Accuracy: 8107/10000 (81.07%)

EPOCH: 54
Loss=0.9632316827774048 Batch_id=390 Accuracy=67.41: 100%|██████████| 391/391 [00:21<00:00, 18.44it/s]

Test set: Average loss: 0.5470, Accuracy: 8168/10000 (81.68%)

EPOCH: 55
Loss=1.276604413986206 Batch_id=390 Accuracy=67.24: 100%|██████████| 391/391 [00:22<00:00, 17.57it/s]

Test set: Average loss: 0.5850, Accuracy: 8105/10000 (81.05%)

EPOCH: 56
Loss=1.1510831117630005 Batch_id=390 Accuracy=67.52: 100%|██████████| 391/391 [00:21<00:00, 18.27it/s]

Test set: Average loss: 0.5575, Accuracy: 8183/10000 (81.83%)

EPOCH: 57
Loss=1.1082019805908203 Batch_id=390 Accuracy=67.68: 100%|██████████| 391/391 [00:21<00:00, 18.62it/s]

Test set: Average loss: 0.5653, Accuracy: 8111/10000 (81.11%)

EPOCH: 58
Loss=0.8811373710632324 Batch_id=390 Accuracy=67.57: 100%|██████████| 391/391 [00:21<00:00, 18.18it/s]

Test set: Average loss: 0.5478, Accuracy: 8215/10000 (82.15%)

EPOCH: 59
Loss=0.9636799693107605 Batch_id=390 Accuracy=67.99: 100%|██████████| 391/391 [00:21<00:00, 18.45it/s]

Test set: Average loss: 0.5260, Accuracy: 8278/10000 (82.78%)

EPOCH: 60
Loss=0.805916965007782 Batch_id=390 Accuracy=68.09: 100%|██████████| 391/391 [00:24<00:00, 15.99it/s]

Test set: Average loss: 0.5277, Accuracy: 8256/10000 (82.56%)

EPOCH: 61
Loss=0.927221417427063 Batch_id=390 Accuracy=68.46: 100%|██████████| 391/391 [00:22<00:00, 17.44it/s]

Test set: Average loss: 0.5418, Accuracy: 8214/10000 (82.14%)

EPOCH: 62
Loss=0.9538083076477051 Batch_id=390 Accuracy=68.30: 100%|██████████| 391/391 [00:21<00:00, 18.53it/s]

Test set: Average loss: 0.5284, Accuracy: 8254/10000 (82.54%)

EPOCH: 63
Loss=0.9099537134170532 Batch_id=390 Accuracy=68.35: 100%|██████████| 391/391 [00:21<00:00, 18.27it/s]

Test set: Average loss: 0.5245, Accuracy: 8279/10000 (82.79%)

EPOCH: 64
Loss=1.1318533420562744 Batch_id=390 Accuracy=68.73: 100%|██████████| 391/391 [00:21<00:00, 18.08it/s]

Test set: Average loss: 0.5178, Accuracy: 8309/10000 (83.09%)

EPOCH: 65
Loss=0.8324493169784546 Batch_id=390 Accuracy=68.39: 100%|██████████| 391/391 [00:21<00:00, 18.24it/s]

Test set: Average loss: 0.5215, Accuracy: 8282/10000 (82.82%)

EPOCH: 66
Loss=0.8703789710998535 Batch_id=390 Accuracy=68.66: 100%|██████████| 391/391 [00:21<00:00, 17.95it/s]

Test set: Average loss: 0.5146, Accuracy: 8264/10000 (82.64%)

EPOCH: 67
Loss=1.0485937595367432 Batch_id=390 Accuracy=68.63: 100%|██████████| 391/391 [00:21<00:00, 18.31it/s]

Test set: Average loss: 0.5008, Accuracy: 8332/10000 (83.32%)

EPOCH: 68
Loss=1.049951434135437 Batch_id=390 Accuracy=68.74: 100%|██████████| 391/391 [00:24<00:00, 15.74it/s]

Test set: Average loss: 0.5244, Accuracy: 8240/10000 (82.40%)

EPOCH: 69
Loss=0.9873534440994263 Batch_id=390 Accuracy=68.29: 100%|██████████| 391/391 [00:21<00:00, 18.29it/s]

Test set: Average loss: 0.5293, Accuracy: 8256/10000 (82.56%)

EPOCH: 70
Loss=0.9089479446411133 Batch_id=390 Accuracy=68.58: 100%|██████████| 391/391 [00:21<00:00, 17.96it/s]

Test set: Average loss: 0.5252, Accuracy: 8267/10000 (82.67%)

EPOCH: 71
Loss=0.9753127098083496 Batch_id=390 Accuracy=68.88: 100%|██████████| 391/391 [00:21<00:00, 18.33it/s]

Test set: Average loss: 0.5050, Accuracy: 8375/10000 (83.75%)

EPOCH: 72
Loss=1.0230995416641235 Batch_id=390 Accuracy=69.23: 100%|██████████| 391/391 [00:21<00:00, 17.83it/s]

Test set: Average loss: 0.5070, Accuracy: 8317/10000 (83.17%)

EPOCH: 73
Loss=0.9421344995498657 Batch_id=390 Accuracy=69.00: 100%|██████████| 391/391 [00:21<00:00, 17.82it/s]

Test set: Average loss: 0.5174, Accuracy: 8314/10000 (83.14%)

EPOCH: 74
Loss=0.9603129625320435 Batch_id=390 Accuracy=68.87: 100%|██████████| 391/391 [00:21<00:00, 18.09it/s]

Test set: Average loss: 0.4983, Accuracy: 8368/10000 (83.68%)

EPOCH: 75
Loss=0.7494078874588013 Batch_id=390 Accuracy=69.22: 100%|██████████| 391/391 [00:25<00:00, 15.17it/s]

Test set: Average loss: 0.5050, Accuracy: 8348/10000 (83.48%)

EPOCH: 76
Loss=1.045647382736206 Batch_id=390 Accuracy=69.22: 100%|██████████| 391/391 [00:21<00:00, 17.78it/s]

Test set: Average loss: 0.4998, Accuracy: 8383/10000 (83.83%)

EPOCH: 77
Loss=1.2078115940093994 Batch_id=390 Accuracy=69.39: 100%|██████████| 391/391 [00:21<00:00, 18.04it/s]

Test set: Average loss: 0.4959, Accuracy: 8385/10000 (83.85%)

EPOCH: 78
Loss=0.9500932693481445 Batch_id=390 Accuracy=69.54: 100%|██████████| 391/391 [00:22<00:00, 17.23it/s]

Test set: Average loss: 0.4993, Accuracy: 8374/10000 (83.74%)

EPOCH: 79
Loss=1.0732629299163818 Batch_id=390 Accuracy=69.59: 100%|██████████| 391/391 [00:21<00:00, 17.90it/s]

Test set: Average loss: 0.4951, Accuracy: 8384/10000 (83.84%)

EPOCH: 80
Loss=0.899154543876648 Batch_id=390 Accuracy=69.25: 100%|██████████| 391/391 [00:21<00:00, 17.81it/s]

Test set: Average loss: 0.4853, Accuracy: 8394/10000 (83.94%)

EPOCH: 81
Loss=0.8721206784248352 Batch_id=390 Accuracy=69.55: 100%|██████████| 391/391 [00:21<00:00, 18.07it/s]

Test set: Average loss: 0.4978, Accuracy: 8352/10000 (83.52%)

EPOCH: 82
Loss=0.9214686155319214 Batch_id=390 Accuracy=69.56: 100%|██████████| 391/391 [00:24<00:00, 16.17it/s]

Test set: Average loss: 0.4900, Accuracy: 8378/10000 (83.78%)

EPOCH: 83
Loss=0.7561839818954468 Batch_id=390 Accuracy=69.88: 100%|██████████| 391/391 [00:21<00:00, 18.36it/s]

Test set: Average loss: 0.4929, Accuracy: 8379/10000 (83.79%)

EPOCH: 84
Loss=0.8193861842155457 Batch_id=390 Accuracy=69.47: 100%|██████████| 391/391 [00:21<00:00, 17.78it/s]

Test set: Average loss: 0.4944, Accuracy: 8385/10000 (83.85%)

EPOCH: 85
Loss=0.925021767616272 Batch_id=390 Accuracy=69.89: 100%|██████████| 391/391 [00:21<00:00, 17.80it/s]

Test set: Average loss: 0.4877, Accuracy: 8374/10000 (83.74%)

EPOCH: 86
Loss=0.9451456069946289 Batch_id=390 Accuracy=70.08: 100%|██████████| 391/391 [00:21<00:00, 17.98it/s]

Test set: Average loss: 0.4940, Accuracy: 8369/10000 (83.69%)

EPOCH: 87
Loss=1.0098634958267212 Batch_id=390 Accuracy=69.93: 100%|██████████| 391/391 [00:21<00:00, 18.53it/s]

Test set: Average loss: 0.4802, Accuracy: 8429/10000 (84.29%)

EPOCH: 88
Loss=0.7206149101257324 Batch_id=390 Accuracy=70.32: 100%|██████████| 391/391 [00:22<00:00, 17.08it/s]

Test set: Average loss: 0.4847, Accuracy: 8429/10000 (84.29%)

EPOCH: 89
Loss=1.1099587678909302 Batch_id=390 Accuracy=69.89: 100%|██████████| 391/391 [00:21<00:00, 17.92it/s]

Test set: Average loss: 0.4839, Accuracy: 8385/10000 (83.85%)

EPOCH: 90
Loss=0.9867966771125793 Batch_id=390 Accuracy=70.05: 100%|██████████| 391/391 [00:24<00:00, 16.27it/s]

Test set: Average loss: 0.4837, Accuracy: 8390/10000 (83.90%)

EPOCH: 91
Loss=0.9626021385192871 Batch_id=390 Accuracy=70.10: 100%|██████████| 391/391 [00:22<00:00, 17.22it/s]

Test set: Average loss: 0.4745, Accuracy: 8458/10000 (84.58%)

EPOCH: 92
Loss=0.8893787264823914 Batch_id=390 Accuracy=70.18: 100%|██████████| 391/391 [00:21<00:00, 18.35it/s]

Test set: Average loss: 0.4878, Accuracy: 8397/10000 (83.97%)

EPOCH: 93
Loss=0.9327006340026855 Batch_id=390 Accuracy=70.21: 100%|██████████| 391/391 [00:21<00:00, 17.98it/s]

Test set: Average loss: 0.4930, Accuracy: 8382/10000 (83.82%)

EPOCH: 94
Loss=0.9631301164627075 Batch_id=390 Accuracy=70.62: 100%|██████████| 391/391 [00:21<00:00, 18.27it/s]

Test set: Average loss: 0.4827, Accuracy: 8420/10000 (84.20%)

EPOCH: 95
Loss=0.8403726816177368 Batch_id=390 Accuracy=70.47: 100%|██████████| 391/391 [00:21<00:00, 17.85it/s]

Test set: Average loss: 0.4788, Accuracy: 8448/10000 (84.48%)

EPOCH: 96
Loss=0.8984915018081665 Batch_id=390 Accuracy=70.69: 100%|██████████| 391/391 [00:21<00:00, 17.99it/s]

Test set: Average loss: 0.4781, Accuracy: 8455/10000 (84.55%)

EPOCH: 97
Loss=1.0671565532684326 Batch_id=390 Accuracy=70.83: 100%|██████████| 391/391 [00:23<00:00, 16.78it/s]

Test set: Average loss: 0.4766, Accuracy: 8428/10000 (84.28%)

EPOCH: 98
Loss=1.0613961219787598 Batch_id=390 Accuracy=70.64: 100%|██████████| 391/391 [00:21<00:00, 18.29it/s]

Test set: Average loss: 0.4653, Accuracy: 8473/10000 (84.73%)

EPOCH: 99
Loss=0.8889687657356262 Batch_id=390 Accuracy=70.52: 100%|██████████| 391/391 [00:22<00:00, 17.76it/s]

Test set: Average loss: 0.4651, Accuracy: 8465/10000 (84.65%)

EPOCH: 100
Loss=1.0912421941757202 Batch_id=390 Accuracy=70.92: 100%|██████████| 391/391 [00:21<00:00, 18.09it/s]

Test set: Average loss: 0.4700, Accuracy: 8462/10000 (84.62%)

EPOCH: 101
Loss=0.888713538646698 Batch_id=390 Accuracy=70.75: 100%|██████████| 391/391 [00:21<00:00, 18.19it/s]

Test set: Average loss: 0.4691, Accuracy: 8456/10000 (84.56%)

EPOCH: 102
Loss=1.0585877895355225 Batch_id=390 Accuracy=70.75: 100%|██████████| 391/391 [00:22<00:00, 17.33it/s]

Test set: Average loss: 0.4745, Accuracy: 8449/10000 (84.49%)

EPOCH: 103
Loss=0.7282471060752869 Batch_id=390 Accuracy=70.53: 100%|██████████| 391/391 [00:21<00:00, 17.98it/s]

Test set: Average loss: 0.4729, Accuracy: 8456/10000 (84.56%)

EPOCH: 104
Loss=0.8846251368522644 Batch_id=390 Accuracy=70.60: 100%|██████████| 391/391 [00:21<00:00, 18.01it/s]

Test set: Average loss: 0.4651, Accuracy: 8474/10000 (84.74%)

EPOCH: 105
Loss=0.6718360781669617 Batch_id=390 Accuracy=70.76: 100%|██████████| 391/391 [00:23<00:00, 16.44it/s]

Test set: Average loss: 0.4690, Accuracy: 8452/10000 (84.52%)

EPOCH: 106
Loss=0.8491015434265137 Batch_id=390 Accuracy=70.98: 100%|██████████| 391/391 [00:22<00:00, 17.74it/s]

Test set: Average loss: 0.4714, Accuracy: 8486/10000 (84.86%)

EPOCH: 107
Loss=1.0628904104232788 Batch_id=390 Accuracy=70.83: 100%|██████████| 391/391 [00:21<00:00, 18.50it/s]

Test set: Average loss: 0.4708, Accuracy: 8452/10000 (84.52%)

EPOCH: 108
Loss=0.936214804649353 Batch_id=390 Accuracy=71.15: 100%|██████████| 391/391 [00:22<00:00, 17.08it/s]

Test set: Average loss: 0.4554, Accuracy: 8524/10000 (85.24%)

EPOCH: 109
Loss=0.9559423327445984 Batch_id=390 Accuracy=70.93: 100%|██████████| 391/391 [00:21<00:00, 18.30it/s]

Test set: Average loss: 0.4622, Accuracy: 8494/10000 (84.94%)

EPOCH: 110
Loss=1.0573064088821411 Batch_id=390 Accuracy=71.43: 100%|██████████| 391/391 [00:22<00:00, 17.54it/s]

Test set: Average loss: 0.4616, Accuracy: 8484/10000 (84.84%)

EPOCH: 111
Loss=0.9516897201538086 Batch_id=390 Accuracy=71.49: 100%|██████████| 391/391 [00:22<00:00, 17.12it/s]

Test set: Average loss: 0.4615, Accuracy: 8482/10000 (84.82%)

EPOCH: 112
Loss=1.037638783454895 Batch_id=390 Accuracy=71.45: 100%|██████████| 391/391 [00:23<00:00, 16.38it/s]

Test set: Average loss: 0.4594, Accuracy: 8501/10000 (85.01%)

EPOCH: 113
Loss=0.8146859407424927 Batch_id=390 Accuracy=71.31: 100%|██████████| 391/391 [00:21<00:00, 18.00it/s]

Test set: Average loss: 0.4592, Accuracy: 8500/10000 (85.00%)

EPOCH: 114
Loss=0.6821272969245911 Batch_id=390 Accuracy=71.57: 100%|██████████| 391/391 [00:21<00:00, 18.17it/s]

Test set: Average loss: 0.4504, Accuracy: 8521/10000 (85.21%)

EPOCH: 115
Loss=0.9336017370223999 Batch_id=390 Accuracy=71.38: 100%|██████████| 391/391 [00:21<00:00, 18.01it/s]

Test set: Average loss: 0.4600, Accuracy: 8507/10000 (85.07%)

EPOCH: 116
Loss=0.9951318502426147 Batch_id=390 Accuracy=71.77: 100%|██████████| 391/391 [00:22<00:00, 17.38it/s]

Test set: Average loss: 0.4564, Accuracy: 8502/10000 (85.02%)

EPOCH: 117
Loss=1.0247509479522705 Batch_id=390 Accuracy=71.81: 100%|██████████| 391/391 [00:22<00:00, 17.47it/s]

Test set: Average loss: 0.4536, Accuracy: 8503/10000 (85.03%)

EPOCH: 118
Loss=0.8150997161865234 Batch_id=390 Accuracy=71.72: 100%|██████████| 391/391 [00:22<00:00, 17.69it/s]

Test set: Average loss: 0.4563, Accuracy: 8517/10000 (85.17%)

EPOCH: 119
Loss=0.8335493803024292 Batch_id=390 Accuracy=71.50: 100%|██████████| 391/391 [00:22<00:00, 17.76it/s]

Test set: Average loss: 0.4557, Accuracy: 8492/10000 (84.92%)

EPOCH: 120
Loss=0.859405517578125 Batch_id=390 Accuracy=71.55: 100%|██████████| 391/391 [00:22<00:00, 17.10it/s]

Test set: Average loss: 0.4571, Accuracy: 8517/10000 (85.17%)

EPOCH: 121
Loss=0.714469313621521 Batch_id=390 Accuracy=71.72: 100%|██████████| 391/391 [00:21<00:00, 18.00it/s]

Test set: Average loss: 0.4545, Accuracy: 8517/10000 (85.17%)

EPOCH: 122
Loss=0.8887971639633179 Batch_id=390 Accuracy=72.01: 100%|██████████| 391/391 [00:21<00:00, 17.82it/s]

Test set: Average loss: 0.4527, Accuracy: 8521/10000 (85.21%)

EPOCH: 123
Loss=0.7451499700546265 Batch_id=390 Accuracy=71.75: 100%|██████████| 391/391 [00:21<00:00, 18.45it/s]

Test set: Average loss: 0.4535, Accuracy: 8517/10000 (85.17%)

EPOCH: 124
Loss=0.8429659605026245 Batch_id=390 Accuracy=71.75: 100%|██████████| 391/391 [00:22<00:00, 17.32it/s]

Test set: Average loss: 0.4528, Accuracy: 8535/10000 (85.35%)

EPOCH: 125
Loss=0.9815629720687866 Batch_id=390 Accuracy=71.88: 100%|██████████| 391/391 [00:21<00:00, 18.18it/s]

Test set: Average loss: 0.4492, Accuracy: 8549/10000 (85.49%)

EPOCH: 126
Loss=1.054311752319336 Batch_id=390 Accuracy=71.97: 100%|██████████| 391/391 [00:21<00:00, 17.86it/s]

Test set: Average loss: 0.4503, Accuracy: 8520/10000 (85.20%)

EPOCH: 127
Loss=0.754169225692749 Batch_id=390 Accuracy=71.44: 100%|██████████| 391/391 [00:26<00:00, 14.68it/s]

Test set: Average loss: 0.4531, Accuracy: 8501/10000 (85.01%)

EPOCH: 128
Loss=0.8883829116821289 Batch_id=390 Accuracy=71.90: 100%|██████████| 391/391 [00:21<00:00, 18.54it/s]

Test set: Average loss: 0.4544, Accuracy: 8525/10000 (85.25%)

EPOCH: 129
Loss=1.1595121622085571 Batch_id=390 Accuracy=71.90: 100%|██████████| 391/391 [00:21<00:00, 17.85it/s]

Test set: Average loss: 0.4497, Accuracy: 8528/10000 (85.28%)

EPOCH: 130
Loss=0.7378989458084106 Batch_id=390 Accuracy=71.81: 100%|██████████| 391/391 [00:21<00:00, 18.02it/s]

Test set: Average loss: 0.4500, Accuracy: 8531/10000 (85.31%)

EPOCH: 131
Loss=0.9597184062004089 Batch_id=390 Accuracy=71.85: 100%|██████████| 391/391 [00:21<00:00, 18.39it/s]

Test set: Average loss: 0.4454, Accuracy: 8520/10000 (85.20%)

EPOCH: 132
Loss=0.8097864389419556 Batch_id=390 Accuracy=71.78: 100%|██████████| 391/391 [00:22<00:00, 17.48it/s]

Test set: Average loss: 0.4503, Accuracy: 8515/10000 (85.15%)

EPOCH: 133
Loss=0.7333136796951294 Batch_id=390 Accuracy=71.83: 100%|██████████| 391/391 [00:21<00:00, 18.37it/s]

Test set: Average loss: 0.4474, Accuracy: 8517/10000 (85.17%)

EPOCH: 134
Loss=0.5845516920089722 Batch_id=390 Accuracy=71.88: 100%|██████████| 391/391 [00:21<00:00, 17.99it/s]

Test set: Average loss: 0.4493, Accuracy: 8520/10000 (85.20%)

EPOCH: 135
Loss=0.9310816526412964 Batch_id=390 Accuracy=71.74: 100%|██████████| 391/391 [00:22<00:00, 17.16it/s]

Test set: Average loss: 0.4484, Accuracy: 8527/10000 (85.27%)

EPOCH: 136
Loss=0.8202729225158691 Batch_id=390 Accuracy=71.86: 100%|██████████| 391/391 [00:21<00:00, 18.07it/s]

Test set: Average loss: 0.4462, Accuracy: 8549/10000 (85.49%)

EPOCH: 137
Loss=0.7179223895072937 Batch_id=390 Accuracy=71.91: 100%|██████████| 391/391 [00:21<00:00, 18.24it/s]

Test set: Average loss: 0.4456, Accuracy: 8533/10000 (85.33%)

EPOCH: 138
Loss=0.8776065111160278 Batch_id=390 Accuracy=71.45: 100%|██████████| 391/391 [00:21<00:00, 18.18it/s]

Test set: Average loss: 0.4485, Accuracy: 8514/10000 (85.14%)

EPOCH: 139
Loss=0.8912870287895203 Batch_id=390 Accuracy=71.99: 100%|██████████| 391/391 [00:21<00:00, 18.34it/s]

Test set: Average loss: 0.4517, Accuracy: 8518/10000 (85.18%)

EPOCH: 140
Loss=0.861286461353302 Batch_id=390 Accuracy=72.16: 100%|██████████| 391/391 [00:21<00:00, 17.79it/s]

Test set: Average loss: 0.4452, Accuracy: 8519/10000 (85.19%)

EPOCH: 141
Loss=1.0321762561798096 Batch_id=390 Accuracy=72.30: 100%|██████████| 391/391 [00:21<00:00, 17.96it/s]

Test set: Average loss: 0.4471, Accuracy: 8516/10000 (85.16%)

EPOCH: 142
Loss=0.7948952317237854 Batch_id=390 Accuracy=72.00: 100%|██████████| 391/391 [00:24<00:00, 16.22it/s]

Test set: Average loss: 0.4475, Accuracy: 8530/10000 (85.30%)

EPOCH: 143
Loss=0.9874517321586609 Batch_id=390 Accuracy=71.83: 100%|██████████| 391/391 [00:22<00:00, 17.50it/s]

Test set: Average loss: 0.4525, Accuracy: 8513/10000 (85.13%)

EPOCH: 144
Loss=0.7395519018173218 Batch_id=390 Accuracy=71.79: 100%|██████████| 391/391 [00:21<00:00, 17.88it/s]

Test set: Average loss: 0.4483, Accuracy: 8528/10000 (85.28%)

EPOCH: 145
Loss=0.9174309968948364 Batch_id=390 Accuracy=71.88: 100%|██████████| 391/391 [00:21<00:00, 18.11it/s]

Test set: Average loss: 0.4419, Accuracy: 8548/10000 (85.48%)

EPOCH: 146
Loss=0.7312885522842407 Batch_id=390 Accuracy=72.02: 100%|██████████| 391/391 [00:22<00:00, 17.11it/s]

Test set: Average loss: 0.4460, Accuracy: 8542/10000 (85.42%)

EPOCH: 147
Loss=0.9412795901298523 Batch_id=390 Accuracy=71.93: 100%|██████████| 391/391 [00:21<00:00, 17.95it/s]

Test set: Average loss: 0.4491, Accuracy: 8531/10000 (85.31%)

EPOCH: 148
Loss=0.8292037844657898 Batch_id=390 Accuracy=72.16: 100%|██████████| 391/391 [00:21<00:00, 17.86it/s]

Test set: Average loss: 0.4474, Accuracy: 8531/10000 (85.31%)

EPOCH: 149
Loss=0.954598069190979 Batch_id=390 Accuracy=71.56: 100%|██████████| 391/391 [00:22<00:00, 17.28it/s]

Test set: Average loss: 0.4472, Accuracy: 8524/10000 (85.24%)

EPOCH: 150
Loss=0.8123047947883606 Batch_id=390 Accuracy=72.11: 100%|██████████| 391/391 [00:22<00:00, 17.73it/s]

Test set: Average loss: 0.4487, Accuracy: 8535/10000 (85.35%)

```

'''
This is my implementation of ResNet in PyTorch.

The paper I am using for reference can be found here: https://arxiv.org/pdf/1512.03385.pdf

From reading the paper, there is one important class to implement. That is Block class that has 3x3 kernel, 64/128/256/512
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
BasicBlock class

Replicating it how the paper described it.

in_channels is to indicate what the input channel size is
out_channels is for the rest of the operations
'''
class BasicResidualBlock(nn.Module):
    '''
    Note:
    we need to include an expansion term to make it easier to plug and play different block types when constructing the resnet
    Notice: this expansion is 1 becase in_channels will always equal out_channels, but in the bottleneck version the number of
    out_channels is 4*in_channels
    '''
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, upsample_shortcut=None):
        super(BasicResidualBlock, self).__init__()
        # expansion value (1 for basic)
        # conv1 -> bn1 -> relu1 -> conv2 -> bn2 -> relu2

        # variable stride in case we are downsampling (maxpooling)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) 
        # stride is 1 because this is the second layer
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
        self.upsample_shortcut = upsample_shortcut
        
    # forward pass function, use functional to implement relu
    def forward(self, x):
        shortcut = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # if we have to upsample the shortcut (defined in resnet class)
        if self.upsample_shortcut is not None:
            shortcut = self.upsample_shortcut(shortcut)

        out += shortcut
        out = F.relu(out)

        return out

'''
The bottlneck functions essentially the same way as the basic residual block.
The key differences is 1) There are three convolutions instead of 2 and we expand the channels by 4 for the last convolution
                       2) The kernels are 1x1 except for the middle convolution which is 3x3
'''
class BottleneckResidualBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, upsample_shortcut=None):
        super(BottleneckResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.upsample_shortcut = upsample_shortcut
    
    def forward(self, x):
        shortcut = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.upsample_shortcut is not None:
            shortcut = self.upsample_shortcut(shortcut)
        
        out += shortcut
        out = F.relu(out)

        return out

'''
ResNet Class.

Note: block_type is used to specify either of the two types of blocks the paper mentioned (BasicBlock and BottleneckBlock)
      num_blocks is an array that we can pass to build the different versions of ResNet without having to explicitly define each one
      num_classes is set to 10 because this is strictly being used for cifar10 dataset
'''
class ResNet(nn.Module):
    def __init__(self, block_type, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(64)

        # contruct the residual blocks (always 4 blocks in paper with varying numbers inside)
        # always in the order of [64, 128, 256, 512]
        # Also note that after the first block the stride changes to 2 for the intial inputs
        self.resblock1 = self.make_ResBlock(block_type, num_blocks[0], 64, stride=1)
        self.resblock2 = self.make_ResBlock(block_type, num_blocks[1], 128, stride=2)
        self.resblock3 = self.make_ResBlock(block_type, num_blocks[2], 256, stride=2)
        self.resblock4 = self.make_ResBlock(block_type, num_blocks[3], 512, stride=2)
        self.linear = nn.Linear(512*block_type.expansion, 10)


    # function to make each layer
    def make_ResBlock(self, block_type, num_blocks, out_channels, stride=1):
        upsample_shortcut = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * block_type.expansion:
            upsample_shortcut = nn.Sequential(nn.Conv2d(self.in_channels, out_channels * block_type.expansion, kernel_size = 1, stride=stride, bias=False),
                                                nn.BatchNorm2d(out_channels * block_type.expansion))
        
        # always start with in_channels and update using expansion channels
        layers.append(block_type(self.in_channels, out_channels, stride=stride, upsample_shortcut=upsample_shortcut))
        self.in_channels = out_channels * block_type.expansion

        # loop through the number of this block we want and create layers
        for i in range(1, num_blocks ):
            layers.append(block_type(self.in_channels, out_channels, stride=1, upsample_shortcut=None))
            # update in_channels
            self.in_channels = out_channels * block_type.expansion
        
        # unpack the list of layers into a nn module 
        return nn.Sequential(*layers)

    # forward function
    def forward(self, x):
        # inital conv
        out = F.relu(self.bn1(self.conv1(x)))
        # res blocks
        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.resblock3(out)
        out = self.resblock4(out)
        out = F.avg_pool2d(out, 4)
        # flatten
        out= out.view(out.size(0), -1)
        # fc layer
        out = self.linear(out)

        return out

# Define the different versions of ResNet
def ResNet18():
    return ResNet(BasicResidualBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicResidualBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(BottleneckResidualBlock, [3, 4, 6, 3])


def ResNet101():
    return ResNet(BottleneckResidualBlock, [3, 4, 23, 3])


def ResNet152():
    return ResNet(BottleneckResidualBlock, [3, 8, 36, 3])
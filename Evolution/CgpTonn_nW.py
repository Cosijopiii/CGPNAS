import sys
from collections import OrderedDict
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class CgpNetW(nn.Module):
    """
    Neural Network class decoding the CGP chromosome.

    Attributes:
        netCGP (list): List of CGP nodes.
        pool_size (int): Pooling size.
        AuxHead (bool): Whether to use auxiliary head.
        arch (OrderedDict): Architecture of the network.
        encode (list): List of encoding layers.
        channel_num (list): List of channel numbers for each layer.
        size (list): List of sizes for each layer.
        layer_module (nn.ModuleList): List of layers in the network.
        outputs (list): List of outputs for each layer.
    """

    def __init__(self, netCGP, in_channel, n_class, imgSize, AuxHead=False):
        """
        Initializes the CgpNetW class.

        Args:
            netCGP (list): List of CGP nodes defining the network architecture.
            in_channel (int): Number of input channels.
            n_class (int): Number of output classes.
            imgSize (int): Size of the input image (assumes square dimensions).
            AuxHead (bool): Whether to include an auxiliary head for intermediate outputs.
        """
        super(CgpNetW, self).__init__()

        self.netCGP = netCGP
        self.AuxHead = AuxHead
        self.encode = []
        self.channel_num = [None] * 2000
        self.size = [None] * 2000

        # Initialize the input layer dimensions
        self.channel_num[0] = in_channel
        self.size[0] = imgSize

        # Mapping for layer types to corresponding classes and parameters
        layer_mappings = {
            'Max_Pool': (nn.MaxPool2d, {'kernel': 2, 'stride': 2}),
            'Avg_Pool': (nn.AvgPool2d, {'kernel': 2, 'stride': 2}),
            'G_Avg_pool': (nn.AdaptiveAvgPool2d, {'output_size': 1}),
            'Indentity': (nn.Identity, {}),
            'AHead': (AuxiliaryHeadCIFAR, {'in_size': None, 'num_classes': n_class}),
            'Concat': (Concat, {}),
            'Sum': (Sum, {}),
            'ConvBlock': (ConvBlock, {'in_size': None, 'out_size': None, 'kernel': None, 'stride': 1}),
            'Bottleneck': (Bottleneck, {'in_size': None, 'out_size': None, 'kernel': None, 'stride': 1}),
            'ResBlock': (ResBlock, {'in_size': None, 'out_size': None, 'kernel': None, 'stride': 1}),
            'SepConv': (SepConv, {'in_size': None, 'out_size': None, 'kernel': None, 'stride': 1, 'padding': 1}),
            'Se': (SE_Block, {'in_size': None}),
            'DiConv': (DilConv, {'in_size': None, 'out_size': None, 'kernel': None, 'stride': 1, 'padding': 2,
                                 'dilation': 2}),
            'C1x7x7x1': (Conv1x7Then7x1, {'in_size': None, 'stride': 1}),
            'MBconv': (MBconv, {'in_size': None, 'out_size': None, 'kernel': None, 'stride': 1}),
            'FusedMBconv': (FusedMBconv, {'in_size': None, 'out_size': None, 'kernel': None, 'stride': 1})
        }

        # Iterate through the CGP nodes to build the architecture
        for i, (name, in1, in2, in3, in4) in enumerate(netCGP):
            if 'input' in name:
                # Skip input nodes
                continue

            if name == 'full':
                # Fully connected layer
                self.encode.append(nn.Linear(self.channel_num[in1] * self.size[in1] ** 2, n_class))
                continue
            if name == 'Max_Pool' or name == 'Avg_Pool':
                self.channel_num[i] = self.channel_num[in1]
                self.size[i] = self.size[in1] // 2
                key = name.split('_')
                func = key[0]
                if func == 'Max':
                    self.encode.append(nn.MaxPool2d(2, stride=2))
                else:
                    self.encode.append(nn.AvgPool2d(2, stride=2))
                continue

            if name == 'G_Avg_pool':
                self.channel_num[i] = self.channel_num[in1]
                self.size[i] = 1
                self.encode.append(nn.AdaptiveAvgPool2d(1))
                continue

            if name == "Indentity":
                self.channel_num[i] = self.channel_num[in1]
                self.size[i] = self.size[in1]
                self.encode.append(nn.Identity())
                continue

            # Handle Sum and Concat separately due to their unique size updates
            if name == 'Sum':
                # Determine the output channels and size based on inputs
                self.channel_num[i] = max(self.channel_num[in1], self.channel_num[in2])
                self.size[i] = min(self.size[in1], self.size[in2])

                # Add the Sum layer
                self.encode.append(Sum())
                continue

            if name == 'Concat':
                # Sum the channels from both inputs
                self.channel_num[i] = self.channel_num[in1] + self.channel_num[in2]

                # Determine the smaller spatial size for alignment
                small_in_id = min(in1, in2, key=lambda idx: self.size[idx])
                self.size[i] = self.size[small_in_id]

                # Add the Concat layer
                self.encode.append(Concat())
                continue

            # Get the base name and parameters for the layer
            # layer_type = name.split('_')[0]
            layer_class, params = layer_mappings.get(name, (None, None))

            if not layer_class:
                raise ValueError(f"Unsupported layer type: {name}")

            # Update parameters dynamically

            if 'in_size' in params:
                params['in_size']= self.channel_num[in1]
            if 'out_size' in params:
                params['out_size'] = in4
            if 'kernel' in params:
                if name!='Max_Pool' and name!='Avg_Pool':
                    params['kernel'] = in3

            # Update the output dimensions for pooling or convolutions
            self.channel_num[i] = params.get('out_size', self.channel_num[in1])
            self.size[i] =  self.size[in1]

            # Add the layer
            self.encode.append(layer_class(**params))

        # Create the module list for all layers
        self.layer_module = nn.ModuleList(self.encode)
        self.outputs = [None] * len(self.netCGP)

    def main(self, x):
        """
        Main function to process the input through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Output tensor and auxiliary output (if any).
        """
        outputs = self.outputs
        outputs[0] = x  # Input image
        nodeID = 1
        aux_output = None

        # Define layer processing functions
        layer_functions = {
            ConvBlock: lambda layer, idx: layer(outputs[self.netCGP[idx][1]]),
            Bottleneck: lambda layer, idx: layer(outputs[self.netCGP[idx][1]], outputs[self.netCGP[idx][1]]),
            ResBlock: lambda layer, idx: layer(outputs[self.netCGP[idx][1]], outputs[self.netCGP[idx][1]]),
            SepConv: lambda layer, idx: layer(outputs[self.netCGP[idx][1]]),
            DilConv: lambda layer, idx: layer(outputs[self.netCGP[idx][1]]),
            MBconv: lambda layer, idx: layer(outputs[self.netCGP[idx][1]], outputs[self.netCGP[idx][1]]),
            FusedMBconv: lambda layer, idx: layer(outputs[self.netCGP[idx][1]], outputs[self.netCGP[idx][1]]),
            SE_Block: lambda layer, idx: layer(outputs[self.netCGP[idx][1]]),
            AuxiliaryHeadCIFAR: lambda layer, idx: layer(outputs[self.netCGP[idx][1]]),
            Conv1x7Then7x1: lambda layer, idx: layer(outputs[self.netCGP[idx][1]]),
            nn.AdaptiveAvgPool2d: lambda layer, idx: layer(outputs[self.netCGP[idx][1]]),
            nn.Linear: lambda layer, idx: layer(
                outputs[self.netCGP[idx][1]].view(outputs[self.netCGP[idx][1]].size(0), -1)),
            nn.MaxPool2d: lambda layer, idx: layer(outputs[self.netCGP[idx][1]]) if outputs[self.netCGP[idx][1]].size(
                2) > 1 else outputs[self.netCGP[idx][1]],
            nn.AvgPool2d: lambda layer, idx: layer(outputs[self.netCGP[idx][1]]) if outputs[self.netCGP[idx][1]].size(
                2) > 1 else outputs[self.netCGP[idx][1]],
            nn.Identity: lambda layer, idx: layer(outputs[self.netCGP[idx][1]]),
            Concat: lambda layer, idx: layer(outputs[self.netCGP[idx][1]], outputs[self.netCGP[idx][2]]),
            Sum: lambda layer, idx: layer(outputs[self.netCGP[idx][1]], outputs[self.netCGP[idx][2]])
        }

        # Process each layer
        for layer in self.layer_module:
            layer_type = type(layer)
            if layer_type in layer_functions:
                # Apply the layer processing function
                outputs[nodeID] = layer_functions[layer_type](layer, nodeID)
                # Capture auxiliary output if applicable
                if layer_type == AuxiliaryHeadCIFAR:
                    aux_output = outputs[nodeID]
            else:
                sys.exit(f"Error at CGP2CNN forward: Unsupported layer type {layer_type}.")
            nodeID += 1

        return outputs[nodeID - 1], aux_output


    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Output tensor and auxiliary output (if any).
        """
        # Process the input through the main network
        Out1, Out2 = self.main(x)
        return Out1, Out2

class AuxiliaryHeadCIFAR(nn.Module):
    """
    Auxiliary Head for CIFAR-like datasets designed to add a branch for auxiliary loss during training.

    This head processes intermediate feature maps to compute auxiliary predictions, which can improve
    training by adding a regularization effect.

    Args:
        in_size (int): Number of input channels.
        num_classes (int): Number of output classes.

    Forward Input:
        x (Tensor): Input tensor of shape (batch_size, in_size, height, width).

    Forward Output:
        Tensor: Output tensor of shape (batch_size, num_classes).
    """

    def __init__(self, in_size, num_classes):
        """Initialize the auxiliary head with feature extraction layers and a classifier."""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # Reduces spatial size
            nn.Conv2d(in_size, 128, 1, bias=False),  # Pointwise convolution to reduce channels
            nn.BatchNorm2d(128),  # Normalizes output from the previous layer
            nn.ReLU(inplace=True),  # Adds non-linearity
            nn.Conv2d(128, 768, 2, bias=False),  # Further increase channels
            nn.BatchNorm2d(768),  # Normalizes the expanded feature map
            nn.ReLU(inplace=True)  # Adds non-linearity
        )
        self.classifier = nn.Linear(768, num_classes)  # Maps to the desired number of classes

    def forward(self, x):
        """
        Perform a forward pass through the auxiliary head.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_size, height, width).

        Returns:
            Tensor: Predicted logits of shape (batch_size, num_classes).
        """
        x = self.features(x)  # Apply feature extraction layers
        x = self.classifier(x.view(x.size(0), -1))  # Flatten and apply the classifier
        return x



class ConvBlock(nn.Module):
    """
    Convolutional Block.

    This block consists of a convolutional layer followed by batch normalization and a ReLU activation.

    Attributes:
        conv (nn.Sequential): Sequential container for the convolutional layer, batch normalization, and ReLU activation.
    """
    def __init__(self, in_size, out_size, kernel=3, stride=1):
        """
        Initializes the ConvBlock class.

        Args:
            in_size (int): Number of input channels.
            out_size (int): Number of output channels.
            kernel (int): Size of the convolutional kernel. Default is 3.
            stride (int): Stride of the convolution. Default is 1.
        """
        super(ConvBlock, self).__init__()
        pad_size = 0 if kernel == 1 else kernel // 2  # Calculate padding size
        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel, stride=stride, padding=pad_size, bias=False),  # Convolutional layer
            nn.BatchNorm2d(out_size),  # Batch normalization
            nn.ReLU(inplace=False)  # ReLU activation
        )

    def forward(self, inputs):
        """
        Forward pass of the convolutional block.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the convolutional block.
        """
        outputs = self.conv(inputs)  # Apply the convolutional block
        return outputs
class DilConv(nn.Module):
    """
    Dilated Convolutional Block.

    This block consists of a depthwise separable convolution followed by batch normalization and a ReLU activation.

    Attributes:
        op (nn.Sequential): Sequential container for the dilated convolutional operations.
    """
    def __init__(self, in_size, out_size, kernel, stride, padding, dilation, affine=True):
        """
        Initializes the DilConv class.

        Args:
            in_size (int): Number of input channels.
            out_size (int): Number of output channels.
            kernel (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding size for the convolution.
            dilation (int): Dilation rate for the convolution.
            affine (bool): Whether to use affine transformation in batch normalization. Default is True.
        """
        super(DilConv, self).__init__()
        pad = 2 if kernel == 3 else 4  # Calculate padding size based on kernel size
        self.op = nn.Sequential(
            nn.Conv2d(in_size, in_size, kernel, stride=stride, padding=pad, dilation=dilation,
                      groups=in_size, bias=False),  # Depthwise convolution
            nn.Conv2d(in_size, out_size,1, padding=0, bias=False),  # Pointwise convolution
            nn.BatchNorm2d(out_size, affine=affine),  # Batch normalization
            nn.ReLU(inplace=False)  # ReLU activation
        )

    def forward(self, x):
        """
        Forward pass of the dilated convolutional block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the dilated convolutional block.
        """
        return self.op(x)  # Apply the dilated convolutional block


class Conv1x7Then7x1(nn.Module):
    """
    A PyTorch module that performs a sequence of convolutions:
    1. A 1x7 convolution followed by
    2. A 7x1 convolution.

    This module is useful for capturing spatial dependencies in both horizontal and vertical directions separately.

    Attributes:
        op (nn.Sequential): The sequential container holding the convolutional operations, batch normalization, and ReLU activation.
    """

    def __init__(self, in_size, stride, affine=True):
        """
        Initializes the Conv1x7Then7x1 module.

        Args:
            in_size (int): The number of input channels.
            stride (int): The stride for the convolutions.
            affine (bool): Whether to use affine parameters in batch normalization. Default is True.
        """
        super(Conv1x7Then7x1, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_size, in_size, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
            nn.Conv2d(in_size, in_size, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(in_size, affine=affine),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        """
        Defines the forward pass of the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the sequence of operations.
        """
        return self.op(x)


class SepConv(nn.Module):
    """
    A PyTorch module that performs a depthwise separable convolution:
    1. A depthwise convolution followed by
    2. A pointwise convolution.

    This module is useful for reducing the number of parameters and computational cost in convolutional neural networks.

    Attributes:
        op (nn.Sequential): The sequential container holding the convolutional operations, batch normalization, and ReLU activation.
    """

    def __init__(self, in_size, out_size, kernel, stride, padding, affine=True):
        """
        Initializes the SepConv module.

        Args:
            in_size (int): The number of input channels.
            out_size (int): The number of output channels.
            kernel (int): The size of the convolutional kernel.
            stride (int): The stride for the convolutions.
            padding (int): The padding for the convolutions.
            affine (bool): Whether to use affine parameters in batch normalization. Default is True.
        """
        super(SepConv, self).__init__()

        pad = kernel // 2

        self.op = nn.Sequential(
            nn.Conv2d(in_size, in_size, kernel, stride=stride, padding=pad, groups=in_size,
                      bias=False),
            nn.Conv2d(in_size, out_size, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_size, affine=affine),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        """
        Defines the forward pass of the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the sequence of operations.
        """
        return self.op(x)


import torch.nn as nn


class SE_Block(nn.Module):
    """
    A PyTorch module implementing a Squeeze-and-Excitation (SE) block.

    This block is designed to recalibrate channel-wise feature responses by explicitly modeling interdependencies between channels.

    Attributes:
        squeeze (nn.AdaptiveAvgPool2d): The adaptive average pooling layer to squeeze the spatial dimensions.
        excitation (nn.Sequential): The sequential container holding the fully connected layers and activation functions for the excitation operation.
    """

    def __init__(self, c, r=4):
        """
        Initializes the SE_Block module.

        Args:
            c (int): The number of input channels.
            r (int): The reduction ratio for the excitation operation. Default is 4.
        """
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Defines the forward pass of the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the squeeze and excitation operations.
        """
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y


import torch
import torch.nn as nn


class MBconv(nn.Module):
    """
    A PyTorch module implementing a Mobile Inverted Bottleneck Convolution (MBConv) block.

    This block is designed to efficiently capture spatial features with depthwise separable convolutions and includes a Squeeze-and-Excitation (SE) block for channel-wise attention.

    Attributes:
        relu (nn.ReLU): The ReLU activation function.
        op (nn.Sequential): The sequential container holding the convolutional operations, batch normalization, ReLU activation, and SE block.
        conv2 (nn.Sequential): The sequential container for the shortcut connection.
    """

    def __init__(self, in_size, out_size, kernel, stride=1, expansion=1):
        """
        Initializes the MBconv module.

        Args:
            in_size (int): The number of input channels.
            out_size (int): The number of output channels.
            kernel (int): The size of the convolutional kernel.
            stride (int): The stride for the convolutions. Default is 1.
            expansion (int): The expansion factor for the intermediate channels. Default is 1.
        """
        super(MBconv, self).__init__()

        if 16 <= out_size < 64:
            expansion = 1
        if 64 <= out_size < 128:
            expansion = 4
        if 128 <= out_size < 256:
            expansion = 4
        if out_size > 256:
            expansion = 4

        pad_size = kernel // 2
        tempOut = out_size * expansion

        self.relu = nn.ReLU(inplace=False)
        self.op = nn.Sequential(
            nn.Conv2d(in_size, tempOut, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(tempOut),
            nn.ReLU(inplace=False),
            nn.Conv2d(tempOut, tempOut,  kernel, stride=1, padding=pad_size, groups=tempOut, bias=False),
            nn.BatchNorm2d(tempOut),
            nn.ReLU(inplace=False),
            SE_Block(tempOut),
            nn.Conv2d(tempOut, out_size, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_size)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_size, out_size, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_size)
        )

    def forward(self, inputs1, inputs2):
        """
        Defines the forward pass of the module.

        Args:
            inputs1 (torch.Tensor): The primary input tensor.
            inputs2 (torch.Tensor): The secondary input tensor for the shortcut connection.

        Returns:
            torch.Tensor: The output tensor after applying the MBConv operations and adding the shortcut connection.
        """
        x = self.op(inputs1)
        x2 = self.conv2(inputs2)
        out = torch.add(x, x2)
        rout = out

        return self.relu(rout)


class MBconv(nn.Module):
    """
    A PyTorch module implementing a Mobile Inverted Bottleneck Convolution (MBConv) block.

    This block is designed to efficiently capture spatial features with depthwise separable convolutions and includes a Squeeze-and-Excitation (SE) block for channel-wise attention.

    Attributes:
        relu (nn.ReLU): The ReLU activation function.
        op (nn.Sequential): The sequential container holding the convolutional operations, batch normalization, ReLU activation, and SE block.
        shortcut (nn.Sequential): The sequential container for the shortcut connection.
    """

    def __init__(self, in_size, out_size, kernel, stride=1, expansion=1):
        """
        Initializes the MBconv module.

        Args:
            in_size (int): The number of input channels.
            out_size (int): The number of output channels.
            kernel (int): The size of the convolutional kernel.
            stride (int): The stride for the convolutions. Default is 1.
            expansion (int): The expansion factor for the intermediate channels. Default is 1.
        """
        super(MBconv, self).__init__()



        if out_size >= 64:
            expansion = 4

        pad_size = kernel // 2
        expanded_channels = out_size * expansion

        self.relu = nn.ReLU(inplace=False)
        self.op = nn.Sequential(

            nn.Conv2d(in_size, expanded_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(expanded_channels, expanded_channels, kernel, stride=1, padding=pad_size,
                      groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU(inplace=False),
            SE_Block(expanded_channels),
            nn.Conv2d(expanded_channels, out_size, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_size)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_size, out_size, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_size)
        )

    def forward(self, inputs1, inputs2):
        """
        Defines the forward pass of the module.

        Args:
            inputs1 (torch.Tensor): The primary input tensor.
            inputs2 (torch.Tensor): The secondary input tensor for the shortcut connection.

        Returns:
            torch.Tensor: The output tensor after applying the MBConv operations and adding the shortcut connection.
        """
        x = self.op(inputs1)
        x2 = self.shortcut(inputs2)
        out = torch.add(x, x2)
        return self.relu(out)


class FusedMBconv(nn.Module):
    """
    A PyTorch module implementing a Fused Mobile Inverted Bottleneck Convolution (Fused MBConv) block.

    This block is designed to efficiently capture spatial features with standard convolutions and includes a Squeeze-and-Excitation (SE) block for channel-wise attention.

    Attributes:
        op (nn.Sequential): The sequential container holding the convolutional operations, batch normalization, ReLU activation, and SE block.
        shortcut (nn.Sequential): The sequential container for the shortcut connection.
        relu (nn.ReLU): The ReLU activation function.
    """

    def __init__(self, in_size, out_size, kernel, stride, expansion=1):
        """
        Initializes the FusedMBconv module.

        Args:
            in_size (int): The number of input channels.
            out_size (int): The number of output channels.
            kernel (int): The size of the convolutional kernel.
            stride (int): The stride for the convolutions.
            expansion (int): The expansion factor for the intermediate channels. Default is 1.
        """
        super(FusedMBconv, self).__init__()


        # Simplify expansion logic
        if out_size >= 64:
            expansion = 4

        pad_size = kernel // 2
        expanded_channels = out_size * expansion

        self.op = nn.Sequential(
            nn.Conv2d(in_size, expanded_channels,  kernel, stride=1, padding=pad_size, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU(inplace=False),
            SE_Block(expanded_channels),
            nn.Conv2d(expanded_channels, out_size, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_size)
        )
        self.relu = nn.ReLU(inplace=False)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_size, out_size, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_size)
        )

    def forward(self, inputs1, inputs2):
        """
        Defines the forward pass of the module.

        Args:
            inputs1 (torch.Tensor): The primary input tensor.
            inputs2 (torch.Tensor): The secondary input tensor for the shortcut connection.

        Returns:
            torch.Tensor: The output tensor after applying the Fused MBConv operations and adding the shortcut connection.
        """
        x = self.op(inputs1)
        x2 = self.shortcut(inputs2)
        out = torch.add(x, x2)
        return self.relu(out)


class Bottleneck(nn.Module):
    """
    A PyTorch module implementing a Linear Bottleneck block.

    This block is designed to reduce the number of parameters and computational cost while maintaining performance.

    Attributes:
        conv1 (nn.Sequential): The sequential container holding the first set of convolutional operations, batch normalization, and ReLU activation.
        conv2 (nn.Sequential): The sequential container for the shortcut connection.
        relu (nn.ReLU): The ReLU activation function.
    """

    def __init__(self, in_size, out_size, kernel, stride):
        """
        Initializes the Bottleneck module.

        Args:
            in_size (int): The number of input channels.
            out_size (int): The number of output channels.
            kernel (int): The size of the convolutional kernel.
            stride (int): The stride for the convolutions.
        """
        super(Bottleneck, self).__init__()

        pad_size = kernel // 2
        reduction = 4 if out_size >= 128 else 2
        temp_out = out_size // reduction

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, temp_out, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(temp_out),
            nn.ReLU(inplace=False),
            nn.Conv2d(temp_out, temp_out, kernel, stride=stride, padding=pad_size, bias=False),
            nn.BatchNorm2d(temp_out),
            nn.ReLU(inplace=False),
            nn.Conv2d(temp_out, out_size, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_size)
        )
        self.relu = nn.ReLU(inplace=False)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_size, out_size, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_size)
        )

    def forward(self, inputs1, inputs2):
        """
        Defines the forward pass of the module.

        Args:
            inputs1 (torch.Tensor): The primary input tensor.
            inputs2 (torch.Tensor): The secondary input tensor for the shortcut connection.

        Returns:
            torch.Tensor: The output tensor after applying the Bottleneck operations and adding the shortcut connection.
        """
        x = self.conv1(inputs1)
        x2 = self.conv2(inputs2)
        out = torch.add(x, x2)
        return self.relu(out)


class ResBlock(nn.Module):
    """
    A PyTorch module implementing a Residual Block.

    This block is designed to facilitate the training of deep neural networks by allowing gradients to flow through the network directly.

    Attributes:
        conv1 (nn.Sequential): The sequential container holding the first set of convolutional operations, batch normalization, and ReLU activation.
        conv2 (nn.Sequential): The sequential container for the shortcut connection.
        relu (nn.ReLU): The ReLU activation function.
    """

    def __init__(self, in_size, out_size, kernel, stride):
        """
        Initializes the ResBlock module.

        Args:
            in_size (int): The number of input channels.
            out_size (int): The number of output channels.
            kernel (int): The size of the convolutional kernel.
            stride (int): The stride for the convolutions.
        """
        super(ResBlock, self).__init__()


        pad_size = kernel // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel, stride, padding=pad_size, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_size, out_size,kernel, stride=stride, padding=pad_size, bias=False),
            nn.BatchNorm2d(out_size)
        )

        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_size, out_size, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_size)
        )

    def forward(self, inputs1, inputs2):
        """
        Defines the forward pass of the module.

        Args:
            inputs1 (torch.Tensor): The primary input tensor.
            inputs2 (torch.Tensor): The secondary input tensor for the shortcut connection.

        Returns:
            torch.Tensor: The output tensor after applying the Residual Block operations and adding the shortcut connection.
        """
        x = self.conv1(inputs1)
        x2 = self.conv2(inputs2)
        out = torch.add(x, x2)
        return self.relu(out)


class Sum(nn.Module):
    """
    A module that sums two input tensors, ensuring compatibility in size and channels.

    If the spatial dimensions of the inputs differ, the larger tensor is downsampled.
    If the channel dimensions differ, the smaller tensor is zero-padded to match.
    """
    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, input1, input2):
        """
        Forward pass to sum two tensors with size adjustments if necessary.

        Args:
            input1 (Tensor): First input tensor.
            input2 (Tensor): Second input tensor.

        Returns:
            Tensor: Summed tensor after size and channel adjustments.
        """
        # Adjust spatial dimensions
        in_data = [input1, input2]
        # check of the image size
        if (in_data[0].size(2) - in_data[1].size(2)) != 0:
            small_in_id, large_in_id = (0, 1) if in_data[0].size(2) < in_data[1].size(2) else (1, 0)
            pool_num = math.floor(in_data[large_in_id].size(2) / in_data[small_in_id].size(2))
            for _ in range(pool_num - 1):
                in_data[large_in_id] = F.max_pool2d(in_data[large_in_id], 2, 2, 0)

        # check of the channel size
        small_ch_id, large_ch_id = (0, 1) if in_data[0].size(1) < in_data[1].size(1) else (1, 0)
        offset = int(in_data[large_ch_id].size()[1] - in_data[small_ch_id].size()[1])
        if offset != 0:
            tmp = in_data[large_ch_id].data[:, :offset, :, :]
            tmp = Variable(tmp).clone()
            in_data[small_ch_id] = torch.cat([in_data[small_ch_id], tmp * 0], 1)
        out = torch.add(in_data[0], in_data[1])
        return out


class Concat(nn.Module):
    """
    A module that concatenates two input tensors along the channel dimension,
    ensuring their spatial dimensions match through downsampling if necessary.
    """
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, input1, input2):
        """
        Forward pass to concatenate two tensors with size adjustments if necessary.

        Args:
            input1 (Tensor): First input tensor.
            input2 (Tensor): Second input tensor.

        Returns:
            Tensor: Concatenated tensor along the channel dimension.
        """
        # Ensure spatial dimensions match by downsampling the larger tensor
        if input1.size(2) != input2.size(2):
            if input1.size(2) > input2.size(2):
                input1 = F.max_pool2d(input1, kernel=2, stride=2)
            else:
                input2 = F.max_pool2d(input2, kernel=2, stride=2)

        # Concatenate along the channel dimension
        return torch.cat([input1, input2], dim=1)


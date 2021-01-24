import torch
import torch.nn as nn



def conv3x3(inplanes, planes):
    return nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)


class ConvolutionCreationLayer(nn.Module):
    '''
        This class is for creating a network layer of convolutions. Each convolution will take place using 3x3 conv filter.
        If pooling is also needed, the argument pooling has to be given as True along with the needed channels for convolution.
        arguments:
            inputChan: the number of channels of the input image (type: int)
            chan1: the number of channels needed after first convolution (type: int)
            chan2: the number of channels needed after second convolution (type: int)
            chan3: the number of channels needed after third convolution (type: int) (default: None)
            pooling: if max pooling is needed (type: bool) (default: True)
            last_activation: the name of the activation function in the last convolution ( choose between ReLU and Sigmoid ) (type: string) (default: 'ReLU')
        
        arguments (function forward):
            image: the input image (type: tensor) (size: batch x 3 x W x H)
        return (function forward):
            image: the output image after convolution ( and max pooling if pooling=True ) (type: tensor)
    '''

    def __init__(self, inputChan, chan1, chan2, chan3=None, pooling=True, last_activation='ReLU'):
        super(ConvolutionCreationLayer, self).__init__()
        self.conv1 = conv3x3(inputChan, chan1)
        self.batchNorm1 = nn.BatchNorm2d(chan1)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(chan1, chan2)
        self.batchNorm2 = nn.BatchNorm2d(chan2)
        self.relu2 = nn.ReLU()
        self.last_layer_fall = True if chan3 == None else False
        if not self.last_layer_fall:
            self.conv3 = conv3x3(chan2, chan3)
            self.batchNorm3 = nn.BatchNorm2d(chan3)
            if last_activation == 'ReLU':
                self.relu3 = nn.ReLU()
            elif last_activation == 'Sigmoid':
                self.relu3 = nn.Sigmoid()
        self.pool = pooling
        if self.pool:
            self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, image):
        image = self.conv1(image)  #Convolution 1
        image = self.batchNorm1(image)
        image = self.relu1(image)

        image = self.conv2(image)  #Convolution 2
        image = self.batchNorm2(image)
        image = self.relu2(image)

        if not self.last_layer_fall:
            image = self.conv3(image)  #Convolution 3
            image = self.batchNorm3(image)
            image = self.relu3(image)

        if self.pool:
            image = self.maxPool(image)
        return image


class DomainAdaptationLayer(nn.Module):
    '''
        This class is for the Domain Adaptation Layer. For now, the layer works only in source domain

        arguments (function forward):
            image: the input image (type: tensor) (size: batch x 384 x W x H)
        return (function forward):
            image: the output image after concatenation (type: tensor)
    '''

    def __init__(self):
        super(DomainAdaptationLayer, self).__init__()

    def forward(self, image):
        return torch.cat((image, image), 3)


class Model(nn.Module):
    '''
        This is the wrapper class for the model.
        arguments:
            last_conv_fall: if the last convolution needs to be removed (type: bool) (default: False)

        arguments (function forward):
            image: the input image (type: tensor) (size: batch x W x H x 3)
            needAvgPool: if the avg pooling at the very end is needed ( if last_conv_fall==True, needAvgPool set to False ) (type: bool) (default: True)
        return (function forward):
            output: the output of the last layer (type: tensor)
    '''

    def __init__(self, last_conv_fall=False):
        super(Model, self).__init__()
        self.firstConv = conv3x3(3, 64)
        self.firstLayer = ConvolutionCreationLayer(64, 128, 196, 128)
        self.firstDownsample = nn.AvgPool2d(kernel_size=4, stride=4)
        self.secondLayer = ConvolutionCreationLayer(128, 128, 196, 128)
        self.secondDownsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.thirdLayer = ConvolutionCreationLayer(128, 128, 196, 128)
        self.domainAdaptationLayer = DomainAdaptationLayer()
        self.last_conv_fall = last_conv_fall
        self.fourthLayer = ConvolutionCreationLayer(768, 128, 64, 1 if not self.last_conv_fall else None, pooling=False, last_activation='Sigmoid')
        if self.last_conv_fall:
            self.linear = nn.Linear(64, 2)

    def forward(self, image, needAvgPool=True):
        if self.last_conv_fall:
            needAvgPool = False
        image = self.firstConv(image)
        image = self.firstLayer(image)
        convImage1 = self.firstDownsample(image)
        convImage1 = convImage1.permute(0, 2, 3, 1)
        image = self.secondLayer(image)
        convImage2 = self.secondDownsample(image)
        convImage2 = convImage2.permute(0, 2, 3, 1)
        convImage3 = self.thirdLayer(image)
        convImage3 = convImage3.permute(0, 2, 3, 1)

        convImage = torch.cat((convImage1, convImage2, convImage3), 3)
        convImage = self.domainAdaptationLayer(convImage)

        convImage = convImage.permute(0, 3, 1, 2)
        image = self.fourthLayer(convImage)

        if needAvgPool:
            output = torch.sum(image, dim=(2,3)) / (14*14)
        else:
            output = image
        if self.last_conv_fall:
            output = self.linear(image.permute(0, 2, 3, 1).view(image.shape[0], -1, 64))

        return output





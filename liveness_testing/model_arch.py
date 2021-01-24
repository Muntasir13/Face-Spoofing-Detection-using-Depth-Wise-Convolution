import torch
import torch.nn as nn



def conv3x3(inplanes, planes):
    return nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)


class ConvLayer(nn.Module):
    '''
        This class is for creating a network layer of convolutions. Each convolution will take place using 3x3 conv filter.
        If pooling is also needed, the argument pooling has to be given as True along with the needed channels for convolution.
        arguments:
            inputChan: the number of channels of the input image (type: int)
            chan1: the number of channels needed after first convolution (type: int)
            chan2: the number of channels needed after second convolution (type: int)
            chan3: the number of channels needed after third convolution (type: int)
            pooling: if max pooling is needed (type: bool)
        
        arguments (function forward):
            image: the input image (type: tensor) (size: batch x 3 x W x H)
        return (function forward):
            image: the output image after convolution ( and max pooling if pooling=True ) (type: tensor)
    '''

    def __init__(self, inputChan, chan1, chan2, chan3, pooling=True, last_activation='ReLU'):
        super(ConvLayer, self).__init__()
        self.conv1 = conv3x3(inputChan, chan1)
        self.batchNorm1 = nn.BatchNorm2d(chan1)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(chan1, chan2)
        self.batchNorm2 = nn.BatchNorm2d(chan2)
        self.relu2 = nn.ReLU()
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

        image = self.conv3(image)  #Convolution 3
        image = self.batchNorm3(image)
        image = self.relu3(image)

        if self.pool:
            image = self.maxPool(image)
        return image


class DomainAdaptation(nn.Module):
    '''
        This class is for the Domain Adaptation Layer. For now, the layer works only in source domain

        arguments (function forward):
            image: the input image (type: tensor) (size: batch x 384 x W x H)
        return (function forward):
            image: the output image after concatenation (type: tensor)
    '''

    def __init__(self):
        super(DomainAdaptation, self).__init__()

    def forward(self, image):
        return torch.cat((image, image), 3)


class Model(nn.Module):
    '''
        This is the wrapper class for the model.

        arguments (function forward):
            image: the input image (type: tensor) (size: batch x W x H x 3)
        return (function forward):
            output: the avg of the output of the last layer and after sigmoid activation (type: tensor)
    '''

    def __init__(self):
        super(Model, self).__init__()
        self.firstConv = conv3x3(3, 64)
        self.firstLayer = ConvLayer(64, 128, 196, 128)
        self.firstDownsample = nn.AvgPool2d(kernel_size=4, stride=4)
        self.secondLayer = ConvLayer(128, 128, 196, 128)
        self.secondDownsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.thirdLayer = ConvLayer(128, 128, 196, 128)
        self.domainAdaptationLayer = DomainAdaptation()
        self.fourthLayer = ConvLayer(768, 128, 64, 1, pooling=False, last_activation='Sigmoid')
        # self.sigmoid = nn.Sigmoid()

    def forward(self, image):
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
        # image = self.sigmoid(image)

        output = torch.sum(image, dim=(2,3)) / (14*14)

        return output





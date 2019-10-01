"""
This model file contains some example code for model parallelism.
We will add some codes to support automatic decision making for model parallelism.
(TODO) update this file
    --> add some function for decision making & automatic partitioning

    - last update: 2019.09.30
    - E.Jubilee Yang
"""
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
"""
ResNet 50 implementation (with partial hybrid setting)
"""
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """ 3x3 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """ 1x1 convolution """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):

        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width = 64')

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):

        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class partial_Bottleneck_front(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, norm_layer=None, partial=1):

        super(partial_Bottleneck_front, self).__init__()
        self.partial = partial

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        if partial > 1:
            self.conv2 = conv3x3(width, width, stride, groups)
            self.bn2 = norm_layer(width)
        if partial > 2:
            self.conv3 = conv1x1(width, planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.partial > 1:
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

        if self.partial > 2:
            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

        return out

class partial_Bottleneck_back(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, norm_layer=None, partial=1):

        super(partial_Bottleneck_back, self).__init__()
        self.partial = partial

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups

        if partial <= 1:
            self.conv1 = conv1x1(inplanes, width)
            self.bn1 = norm_layer(width)
        if partial <= 2:
            self.conv2 = conv3x3(width, width, stride, groups)
            self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, out=None):

        identity = x

        if self.partial <= 1:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

        if self.partial <= 2:
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):

        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.__make__layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self.__make__layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self.__make__layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self.__make__layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3%

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def __make__layer(self, block, planes, blocks, stride=1, norm_layer=None):

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNet_hybrid_front(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):

        super(ResNet_hybrid_front, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.__make__layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self.__make__layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self.__make__layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self.__make__layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3%

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def __make__layer(self, block, planes, blocks, stride=1, norm_layer=None):

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class ResNet_hybrid_rear(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):

        super(ResNet_hybrid_rear, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 2048
        self.groups = groups
        self.base_width = width_per_group
        self.layer4 = self.__make__layer(block, 512, layers[0], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3%

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def __make__layer(self, block, planes, blocks, stride=1, norm_layer=None):

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet50(pretrained=False, **kwargs):
    """ Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet(Bottleneck, [3,4,6,3], **kwargs)
    return model

def resnet18(pretrained=False, **kwargs):

    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet50_front(pretrained=False, **kwargs):
    """
    resnet_front only contains conv1~conv5_3

    :param pretrained:
    :param kwargs:
    :return:
    """
    model = ResNet_hybrid_front(Bottleneck, [3,4,6,1], **kwargs)
    return model

def resnet50_rear(pretrained=False, **kwargs):
    """
    resnet_rear only contains conv5_4~conf5_9 + fc

    :param pretrained:
    :param kwargs:
    :return:
    """
    model = ResNet_hybrid_rear(Bottleneck, [2], **kwargs)
    return model


def resnet101(pretrained=False, **kwargs):
    """ Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet(Bottleneck, [3,4,23,3], **kwargs)
    return model


def resnet101_trial1_front(pretrained=False, **kwargs):
    """
    resnet_front only contains conv1~conv5_3

    :param pretrained:
    :param kwargs:
    :return:
    """
    model = ResNet_hybrid_front(Bottleneck, [3,4,23,1], **kwargs)
    return model

def resnet101_trial1_rear(pretrained=False, **kwargs):
    """
    resnet_rear only contains conv5_4~conf5_9 + fc

    :param pretrained:
    :param kwargs:
    :return:
    """
    model = ResNet_hybrid_rear(Bottleneck, [2], **kwargs)
    return model



# [3, 4, 23, 3]
# resnet101_trial2_front [3, 4, 10, 0]
# resnet101_trial2_rear [0, 0, 0, 13, 3] inplanes = 1024

class ResNet_hybrid(nn.Module):

    def __init__(self, block, layers, is_rear=False, inplanes = 64, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):

        super(ResNet_hybrid, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = inplanes
        self.groups = groups
        self.base_width = width_per_group
        self.layers = []
        self.is_rear  = is_rear

        if layers[0] is not 0:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self.__make__layer(block, 64, layers[0], norm_layer=norm_layer)
            self.layers.append(self.conv1)
            self.layers.append(self.bn1)
            self.layers.append(self.relu)
            self.layers.append(self.maxpool)
            self.layers.append(self.layer1)

        if layers[1] is not 0:
            self.layer2 = self.__make__layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
            self.layers.append(self.layer2)

        if layers[2] is not 0:
            self.layer3 = self.__make__layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
            self.layers.append(self.layer3)

        if layers[3] is not 0:
            self.layer4 = self.__make__layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
            self.layers.append(self.layer4)

        if is_rear :
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3%

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def __make__layer(self, block, planes, blocks, stride=1, norm_layer=None):

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):


        for l in self.layers:
            x = l(x)

        if self.is_rear:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)


        return x

def resnet101_trial4_front(pretrained=False, **kwargs):
    """
    resnet_trial4_front only (conv1~conv4_24)
    :param pretrained:
    :param kwargs:
    :return:
    """
    model = ResNet_hybrid(Bottleneck, [3,4,8,0],False,64, **kwargs)
    return model

def resnet101_trial4_rear(pretrained=False, **kwargs):
    """
    resnet101_tiral4_rear only (conv4_25~fc)
    :param pretrained:
    :param kwargs:
    :return:
    """
    model = ResNet_hybrid(Bottleneck, [0,0,15,3],True,1024, **kwargs)
    return model

def resnet101_trial3_front(pretrained=False, **kwargs):
    """
    resnet_trial3_front only (conv1~conv4_9)
    :param pretrained:
    :param kwargs:
    :return:
    """
    model = ResNet_hybrid(Bottleneck, [3,4,3,0], False, 64, **kwargs)
    return model

def resnet101_trial3_rear(pretrained=False, **kwargs):
    """
    resent_trial3_rear only (conv4_10~fc)
    :param pretrained:
    :param kwargs:
    :return:
    """
    model = ResNet_hybrid(Bottleneck, [0, 0, 20, 3], True, 1024, **kwargs)
    return model


def resnet101_trial2_front(pretrained=False, **kwargs):
    """
    resnet_trial2_front only
    :param pretrained:
    :param kwargs:
    :return:
    """
    model = ResNet_hybrid(Bottleneck, [3,4,10,0], False, 64, **kwargs)
    return model

def resnet101_trial2_rear(pretrained=False, **kwargs):
    """

    :param pretrained:
    :param kwargs:
    :return:
    """
    model = ResNet_hybrid(Bottleneck, [0,0,13,3], True, 1024, **kwargs)
    return model





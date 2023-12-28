import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ------------------原论文中的block（始）------------------
# Ordinary UNet Conv Block 卷积块
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm3d(out_size)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_size)
        self.activation = activation

        nn.init.xavier_uniform(self.conv.weight, gain=np.sqrt(2.0))
        nn.init.constant(self.conv.bias, 0)
        nn.init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2.0))
        nn.init.constant(self.conv2.bias, 0)

    def forward(self, x):
        out = self.activation(self.bn(self.conv(x)))
        out = self.activation(self.bn2(self.conv2(out)))

        return out


# two-layer residual unit: two conv with BN/relu and identity mapping 残差单元
class residualUnit(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, stride=1, padding=1, activation=F.relu):
        super(residualUnit, self).__init__()
        self.conv1 = nn.Conv3d(in_size, out_size, kernel_size, stride=1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight, gain=np.sqrt(2.0))  # or gain=1
        nn.init.constant(self.conv1.bias, 0)
        self.conv2 = nn.Conv3d(out_size, out_size, kernel_size, stride=1, padding=1)
        nn.init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2.0))  # or gain=1
        nn.init.constant(self.conv2.bias, 0)
        self.activation = activation
        self.bn1 = nn.BatchNorm3d(out_size)
        self.bn2 = nn.BatchNorm3d(out_size)
        self.in_size = in_size
        self.out_size = out_size
        if in_size != out_size:
            self.convX = nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0)
            self.bnX = nn.BatchNorm3d(out_size)

    def forward(self, x):
        out1 = self.activation(self.bn1(self.conv1(x)))
        out2 = self.activation(self.bn2(self.conv2(out1)))
        if self.in_size != self.out_size:
            bridge = self.activation(self.bnX(self.convX(x)))
        output = torch.add(out2, bridge)

        return output


# Ordinary Residual UNet-Up Conv Block
class UNetUpResBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpResBlock, self).__init__()
        # 转置卷积 输入
        # print("before size:",in_size,out_size)
        self.up = nn.ConvTranspose3d(in_size, out_size, 2, stride=2)  # 为了抑制棋盘效应，改成了(1,1)发现会报错  #7.11改成3,3（依旧报错）
        # print("after size:",in_size,out_size)

        self.bnup = nn.BatchNorm3d(out_size)
        nn.init.xavier_uniform(self.up.weight, gain=np.sqrt(2.0))
        nn.init.constant(self.up.bias, 0)

        self.activation = activation

        self.resUnit = residualUnit(in_size, out_size, kernel_size=kernel_size)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height, layer_depth = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        # print ('    x.shape: ',x.shape, ' \n    bridge.shape: ',bridge.shape)
        up = self.activation(self.bnup(self.up(x)))

        # crop1 = self.center_crop(bridge, up.size()[2])
        # print ('    up.shape: ',up.shape, ' \n    crop1.shape: ',crop1.shape)

        crop1 = bridge
        # print ('    up.shape: ',up.shape, ' \n    crop1.shape: ',crop1.shape)

        out = torch.cat([up, crop1], 1)

        out = self.resUnit(out)
        # out = self.activation(self.bn2(self.conv2(out)))

        return out


class UNetUpResBlock_223(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpResBlock_223, self).__init__()
        # 转置卷积 输入
        # print("before size:",in_size,out_size)
        self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=3, stride=(2, 2, 3), padding=1,
                                     output_padding=(1, 1, 2))
        # (original) kernel_size=2 out_padding=0

        # print("after size:",in_size,out_size)

        self.bnup = nn.BatchNorm3d(out_size)
        nn.init.xavier_uniform(self.up.weight, gain=np.sqrt(2.0))
        nn.init.constant(self.up.bias, 0)

        self.activation = activation

        self.resUnit = residualUnit(in_size, out_size, kernel_size=kernel_size)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height, layer_depth = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        # print ('    x.shape: ',x.shape, ' \n    bridge.shape: ',bridge.shape)
        up = self.activation(self.bnup(self.up(x)))

        # crop1 = self.center_crop(bridge, up.size()[2])
        # print ('    up.shape: ',up.shape, ' \n    crop1.shape: ',crop1.shape)

        crop1 = bridge
        # print ('    up.shape: ',up.shape, ' \n    crop1.shape: ',crop1.shape)

        out = torch.cat([up, crop1], 1)

        out = self.resUnit(out)
        # out = self.activation(self.bn2(self.conv2(out)))

        return out


# ------------------原论文中的block（末）------------------

class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs two convolutions followed by a max pool operation.
    Values:
        input_channels: the number of channels to expect from a given input
    '''

    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super(ContractingBlock, self).__init__()
        # self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        self.conv1 = nn.Conv3d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        if use_bn:
            # self.batchnorm = nn.BatchNorm2d(input_channels * 2)
            self.batchnorm = nn.BatchNorm3d(input_channels * 2)
        self.use_bn = use_bn
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock:
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x


class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a U-Net -
    maps each pixel to a pixel with the correct number of output dimensions
    using a 1x1 convolution.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''

    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        # self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        '''
        Function for completing a forward pass of FeatureMapBlock:
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x


# ------------------原论文中的模型（始）------------------
# 原论文的残差u-net作为生成器
class ResUNet_LRes(nn.Module):
    def __init__(self, in_channel=1, n_classes=4, dp_prob=0):
        super(ResUNet_LRes, self).__init__()
        # self.imsize = imsize

        self.activation = F.relu

        self.pool1 = nn.MaxPool3d(2)
        # self.pool1 = nn.MaxPool3d((3,3,3),(1,1,1),(2,2,3))
        self.pool2 = nn.MaxPool3d(2)
        # self.pool3 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(3, stride=(2, 2, 3), padding=1)
        # self.pool4 = nn.MaxPool3d(2)

        # hidden_channel = 32
        hidden_channel = 16
        self.conv_block1_64 = UNetConvBlock(in_channel, hidden_channel)
        self.conv_block64_128 = residualUnit(hidden_channel, hidden_channel * 2)
        self.conv_block128_256 = residualUnit(hidden_channel * 2, hidden_channel * 4)
        self.conv_block256_512 = residualUnit(hidden_channel * 4, hidden_channel * 8)
        # self.conv_block512_1024 = residualUnit(512, 1024)
        # this kind of symmetric design is awesome, it automatically solves the number of channels during upsamping
        # self.up_block1024_512 = UNetUpResBlock(1024, 512)
        self.up_block512_256 = UNetUpResBlock_223(hidden_channel * 8, hidden_channel * 4)
        self.up_block256_128 = UNetUpResBlock(hidden_channel * 4, hidden_channel * 2)
        self.up_block128_64 = UNetUpResBlock(hidden_channel * 2, hidden_channel)
        self.Dropout = nn.Dropout3d(p=dp_prob)
        self.last = nn.Conv3d(hidden_channel, n_classes, 1, stride=1)

    # def forward(self, x, res_x):
    def forward(self, x):
        res_x = x
        #         print 'line 70 ',x.size()
        block1 = self.conv_block1_64(x)  # (hc,80,112,84)
        # print ('block1.shape: ', block1.shape)
        pool1 = self.pool1(block1)  # (hc,40,56,42)
        # print ('pool1.shape: ', block1.shape)
        pool1_dp = self.Dropout(pool1)  # (hc,40,56,42)
        # print ('pool1_dp.shape: ', pool1_dp.shape)
        block2 = self.conv_block64_128(pool1_dp)  # (hc*2,40,56,42)
        # print ('block2.shape: ', block2.shape)
        pool2 = self.pool2(block2)  # (hc*2,20,28,21)
        # print ('pool2.shape: ', pool2.shape)
        pool2_dp = self.Dropout(pool2)  # (hc*2,20,28,21)
        # print ('pool2_dp.shape: ', pool2_dp.shape)

        block3 = self.conv_block128_256(pool2_dp)  # (hc*4,20,28,21)
        # print ('block3.shape: ', block3.shape)

        pool3 = self.pool3(block3)  # (hc*4,10,14,7)
        # print ('pool3.shape: ', pool3.shape)

        pool3_dp = self.Dropout(pool3)  # (hc*4,10,14,7)
        # print ('pool3_dp.shape: ', pool3_dp.shape)

        block4 = self.conv_block256_512(pool3_dp)  # (hc*8,10,14,7)
        # print ('block4.shape: ', block4.shape)

        # pool4 = self.pool4(block4)
        # pool4_dp = self.Dropout(pool4)
        # # block5 = self.conv_block512_1024(pool4_dp)
        # up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(block4, block3)
        # print ('up2.shape: ', up2.shape)

        up3 = self.up_block256_128(up2, block2)
        # up3 = self.up_block256_128(block3, block2)  #如果不用512-256的话启用这个
        # print ('up3.shape: ', up3.shape)

        up4 = self.up_block128_64(up3, block1)
        # print ('up4.shape: ', up4.shape)

        last = self.last(up4)

        out = last
        # print ('res_x.shape is ', res_x.shape, ' and last.shape is ', last.shape)
        if len(res_x.shape) == 3:
            res_x = res_x.unsqueeze(1)
        out = torch.add(last, res_x)

        # print ('out.shape is ',out.shape)
        return out


# 原论文的论文的的鉴别器（CNN）
'''
    Discriminator for the reconstruction project
'''


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # you can make abbreviations for conv and fc, this is not necessary
        # class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv3d(1, 32, 9)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, 5)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 64, 5)
        self.bn3 = nn.BatchNorm3d(64)
        self.fc1 = nn.Linear(64 * 2 * 11 * 20, 512)
        # self.bn3= nn.BatchNorm1d(6)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        #         print 'line 114: x shape: ',x.size()
        # x = F.max_pool3d(F.relu(self.bn1(self.conv1(x))),(2,2,2))#conv->relu->pool
        x = F.max_pool3d(F.relu(self.conv1(x)), (2, 2, 2))  # conv->relu->pool
        x_1 = x
        x = F.max_pool3d(F.relu(self.conv2(x)), (2, 2, 2))  # conv->relu->pool
        x_2 = x
        x = F.max_pool3d(F.relu(self.conv3(x)), (2, 2, 2))  # conv->relu->pool
        x_3 = x
        # reshape them into Vector, review ruturned tensor shares the same data but have different shape, same as reshape in matlab
        x = x.view(-1, self.num_of_flat_features(x))
        x_4 = x
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        # x = F.sigmoid(x)
        # print 'min,max,mean of x in 0st layer',x.min(),x.max(),x.mean()
        return x

    def num_of_flat_features(self, x):
        size = x.size()[1:]  # we donot consider the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# ------------------原论文中的模型（末）------------------

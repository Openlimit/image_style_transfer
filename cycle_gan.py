from nn_tools import layer
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_begin = layer.Conv(in_channels, 64, 7, padding=3, padding_type='reflect',
                                     with_norm=True, norm_type='instance', activation=nn.ReLU(True), use_bias=True)

        self.down = nn.Sequential(
            layer.Conv(64, 128, 3, stride=2, padding=1, padding_type='reflect',
                       with_norm=True, norm_type='instance', activation=nn.ReLU(True), use_bias=True),
            layer.Conv(128, 256, 3, stride=2, padding=1, padding_type='reflect',
                       with_norm=True, norm_type='instance', activation=nn.ReLU(True), use_bias=True)
        )

        resnet_list = []
        for i in range(6):
            resnet = layer.ResnetBlock(256, 3, padding=1, padding_type='reflect',
                                       with_norm=True, norm_type='instance', activation=nn.ReLU(True), use_bias=True)
            resnet_list.append(resnet)

        self.resnets = nn.Sequential(*resnet_list)

        self.up = nn.Sequential(
            layer.ConvTranspose(256, 128, 3, stride=2, padding=1, output_padding=1,
                                with_norm=True, norm_type='instance', activation=nn.ReLU(True), use_bias=True),
            layer.ConvTranspose(128, 64, 3, stride=2, padding=1, output_padding=1,
                                with_norm=True, norm_type='instance', activation=nn.ReLU(True), use_bias=True)
        )

        self.conv_end = layer.Conv(64, out_channels, 7, padding=3, padding_type='reflect', with_norm=False,
                                   activation=nn.Tanh(), use_bias=True)

    def forward(self, x):
        x = self.conv_begin(x)
        x = self.down(x)
        x = self.resnets(x)
        x = self.up(x)
        x = self.conv_end(x)

        return x

    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad


# Defines the PatchGAN discriminator with the specified arguments.
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.conv1 = layer.Conv(in_channels, 64, 4, stride=2, padding=1, with_norm=False,
                                activation=nn.LeakyReLU(0.2, True))
        self.conv2 = layer.Conv(64, 128, 4, stride=2, padding=1, with_norm=True, norm_type='instance',
                                activation=nn.LeakyReLU(0.2, True), use_bias=True)
        self.conv3 = layer.Conv(128, 256, 4, stride=2, padding=1, with_norm=True, norm_type='instance',
                                activation=nn.LeakyReLU(0.2, True), use_bias=True)
        self.conv4 = layer.Conv(256, 512, 4, stride=1, padding=1, with_norm=True, norm_type='instance',
                                activation=nn.LeakyReLU(0.2, True), use_bias=True)
        self.conv5 = layer.Conv(512, 1, 4, stride=1, padding=1, with_norm=False,
                                activation=None, use_bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x

    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad

import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, padding_type='zero', with_norm=True,
                 norm_type='batch', activation=nn.ReLU(), use_bias=False):
        super(ResnetBlock, self).__init__()

        self.conv1 = Conv(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                          padding_type=padding_type,
                          with_norm=with_norm, norm_type=norm_type, activation=activation, use_bias=use_bias)

        self.conv2 = Conv(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                          padding_type=padding_type,
                          with_norm=with_norm, norm_type=norm_type, activation=None)

        self.conv_block = nn.Sequential(self.conv1, self.conv2)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, padding_type='zero',
                 with_norm=True, norm_type='batch', activation=nn.ReLU(), drop_rate=0.0, use_bias=False):
        super(Conv, self).__init__()

        self.conv = None
        if padding_type == 'zero':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                  bias=use_bias)
        elif padding_type == 'reflect':
            self.conv = nn.Sequential(
                nn.ReflectionPad2d(padding), nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                                                       padding=0, bias=use_bias))
        elif padding_type == 'replicate':
            self.conv = nn.Sequential(
                nn.ReplicationPad2d(padding), nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                                                        padding=0, bias=use_bias))
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.norm = None
        if with_norm:
            if norm_type == 'batch':
                self.norm = BatchNorm(out_channels)
            elif norm_type == 'instance':
                self.norm = InstanceNorm(out_channels)
            else:
                raise NotImplementedError('norm [%s] is not implemented' % norm_type)

        self.activation = activation
        self.drop = Dropout2d(drop_rate) if drop_rate > 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        if self.drop:
            x = self.drop(x)

        return x


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, with_norm=True, norm_type='batch', activation=nn.ReLU(), drop_rate=0.0,
                 use_bias=True):
        super(ConvTranspose, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                       output_padding=output_padding, bias=use_bias)

        self.norm = None
        if with_norm:
            if norm_type == 'batch':
                self.norm = BatchNorm(out_channels)
            elif norm_type == 'instance':
                self.norm = InstanceNorm(out_channels)
            else:
                raise NotImplementedError('norm [%s] is not implemented' % norm_type)

        self.activation = activation
        self.drop = Dropout2d(drop_rate) if drop_rate > 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        if self.drop:
            x = self.drop(x)

        return x


class Dense(nn.Module):
    def __init__(self, in_channels, out_channels, with_norm=False, norm_type='batch', drop_rate=0.0,
                 activation=nn.ReLU()):
        super(Dense, self).__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = None
        if with_norm:
            if norm_type == 'batch':
                self.norm = BatchNorm1d(out_channels)
            elif norm_type == 'instance':
                self.norm = InstanceNorm1d(out_channels)
            else:
                raise NotImplementedError('norm [%s] is not implemented' % norm_type)
        self.activation = activation
        self.drop = Dropout(drop_rate) if drop_rate > 0 else None

    def forward(self, x):
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        if self.drop:
            x = self.drop(x)
        return x


def Dropout(drop_rate):
    return nn.Dropout(drop_rate)


def Dropout2d(drop_rate):
    return nn.Dropout2d(drop_rate)


def BatchNorm(num_features):
    return nn.BatchNorm2d(num_features, affine=True)


def BatchNorm1d(num_features):
    return nn.BatchNorm1d(num_features)


def InstanceNorm(num_features):
    return nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)


def InstanceNorm1d(num_features):
    return nn.InstanceNorm1d(num_features, affine=False, track_running_stats=False)

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn import functional as F  # ？
from Liblinks.sn_lib import *
from torch.nn.utils import spectral_norm


# from Liblinks.utis import *


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=None, ksize=3, pad=1, downsample=False):
        super(Block, self).__init__()
        self.downsample = downsample
        self.skip_connection = (in_channel != out_channel) or downsample
        hidden_channel = in_channel if hidden_channel is None else hidden_channel
        self.residual = nn.Sequential(nn.ReLU(inplace=True),
                                      SNConv2d(in_channel, hidden_channel, kernel_size=ksize, padding=pad),
                                      nn.ReLU(inplace=True),
                                      SNConv2d(hidden_channel, out_channel, kernel_size=ksize, padding=pad))
        self.c1 = SNConv2d(in_channel, hidden_channel, kernel_size=ksize, padding=pad)
        self.c2 = SNConv2d(hidden_channel, out_channel, kernel_size=ksize, padding=pad)
        self.act_func = nn.ReLU(inplace=True)
        if self.skip_connection:
            self.c_sc = SNConv2d(in_channel, out_channel, kernel_size=1, padding=0)  #
        if downsample:
            self._downsample = nn.AvgPool2d(2)

    def residual_down(self, x):
        h = self.residual(x)
        if self.downsample:
            h = self._downsample(h)
        return h

    def shortcut(self, x):
        if self.skip_connection:
            x = self.c_sc(x)
            if self.downsample:
                return self._downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual_down(x) + self.shortcut(x)


class OptimizedBlock(nn.Module):
    def __init__(self, in_channel, out_channel, ksize=3, pad=1):
        super(OptimizedBlock, self).__init__()
        self.residual = nn.Sequential(SNConv2d(in_channel, out_channel, kernel_size=ksize, padding=pad),
                                      nn.ReLU(inplace=True),
                                      SNConv2d(out_channel, out_channel, kernel_size=ksize, padding=pad),
                                      nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(nn.AvgPool2d(2), SNConv2d(in_channel, out_channel, kernel_size=1, padding=0))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class GBlock(nn.Module):  # 生成器的resblock
    def __init__(self, in_channel, out_channel, hidden_channel=None, ksize=3, pad=1, upsample=False, n_cls=0):
        super(GBlock, self).__init__()
        self.upsample = upsample
        hidden_channel = out_channel if hidden_channel is None else hidden_channel
        self.learnable_sc = in_channel != out_channel or upsample
        self.num_classes = n_cls
        self.residual = nn.Sequential(nn.BatchNorm2d(in_channel),
                                      nn.ReLU(inplace=True),
                                      nn.ConvTranspose2d(in_channel, hidden_channel, kernel_size=(4,4), stride=2,
                                                         padding=pad),  # ？
                                      nn.BatchNorm2d(hidden_channel),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(hidden_channel, out_channel, kernel_size=ksize, padding=pad)
                                      )
        if self.learnable_sc:
            if self.upsample:
                #self.c_sc = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=ksize, stride=2, padding=pad,
                #                               output_padding=1)  # ？
                self.c_sc = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(4,4), stride=(2,2), padding=pad)  # ？
            else:
                self.c_sc = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            return self.c_sc(x)
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class GBlockDecon(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=None, ksize=3, pad=1, upsample=False, n_cls=0):
        super(GBlockDecon, self).__init__()
        self.upsample = upsample
        hidden_channel = out_channel if hidden_channel is None else hidden_channel
        self.learnable_sc = in_channel != out_channel or upsample
        self.num_classes = n_cls
        self.residual = nn.Sequential(nn.BatchNorm2d(in_channel),
                                      nn.ReLU(inplace=True),
                                      spectral_norm(nn.ConvTranspose2d(in_channel, hidden_channel, kernel_size=(4,4), stride=2,
                                                         padding=pad)),  # ？
                                      nn.BatchNorm2d(hidden_channel),
                                      nn.ReLU(inplace=True),
                                      spectral_norm(nn.Conv2d(hidden_channel, out_channel, kernel_size=ksize, padding=pad))
                                      )
        if self.learnable_sc:
            if self.upsample:
                self.c_sc = spectral_norm(nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(4,4), stride=2, padding=pad))  # ？
                #self.c_sc = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=1, stride=2, padding=0,
                #                               output_padding=1)  # ？
            else:
                self.c_sc = spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0))

    def shortcut(self, x):
        if self.learnable_sc:
            return self.c_sc(x)
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

def _upsample(x):
    return x


def upsample_conv(x, conv):
    return conv(_upsample(x))


class GBlock2(nn.Module):  # 生成器的resblock2
    def __init__(self, in_channel, out_channel, hidden_channel=None, ksize=(3, 3), pad=(1, 1), upsample=False, n_cls=0,up_mode='bilinear'):
        super(GBlock2, self).__init__()
        self.upsample = upsample
        hidden_channel = out_channel if hidden_channel is None else hidden_channel
        self.learnable_sc = in_channel != out_channel or upsample
        self.num_classes = n_cls

        self.residual = nn.Sequential(nn.BatchNorm2d(in_channel),
                                      nn.ReLU(inplace=True),
                                      # nn.ConvTranspose2d(in_channel, hidden_channel, kernel_size=ksize, stride=2,
                                      #                   padding=pad,
                                      #                   output_padding=1),  # ？
                                      nn.Upsample(scale_factor=2, mode=up_mode),
                                      nn.Conv2d(in_channel, hidden_channel, kernel_size=ksize, padding=pad),
                                      nn.BatchNorm2d(hidden_channel),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(hidden_channel, out_channel, kernel_size=ksize, padding=pad)
                                      )
        if self.learnable_sc:
            if self.upsample:
                self.c_sc = nn.Sequential(nn.Upsample(scale_factor=2, mode=up_mode),
                                          nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0))
                # nn.ConvTranspose2d(in_channel, out_channel, kernel_size=1, stride=2, padding=0,
                #                   output_padding=1)  # ？
            else:
                self.c_sc = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            return self.c_sc(x)
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class SNGBlock(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=None, ksize=(3, 3), pad=(1, 1), upsample=False, n_cls=0,up_mode='bilinear'):
        super(SNGBlock, self).__init__()
        self.upsample = upsample
        hidden_channel = out_channel if hidden_channel is None else hidden_channel
        self.learnable_sc = in_channel != out_channel or upsample
        self.num_classes = n_cls

        self.residual = nn.Sequential(nn.BatchNorm2d(in_channel),
                                      nn.ReLU(inplace=True),
                                      # nn.ConvTranspose2d(in_channel, hidden_channel, kernel_size=ksize, stride=2,
                                      #                   padding=pad,
                                      #                   output_padding=1),  # ？
                                      nn.Upsample(scale_factor=2, mode=up_mode),
                                      spectral_norm(nn.Conv2d(in_channel, hidden_channel, kernel_size=ksize, padding=pad)),
                                      nn.BatchNorm2d(hidden_channel),
                                      nn.ReLU(inplace=True),
                                      spectral_norm(nn.Conv2d(hidden_channel, out_channel, kernel_size=ksize, padding=pad))
                                      )
        if self.learnable_sc:
            if self.upsample:
                self.c_sc = nn.Sequential(nn.Upsample(scale_factor=2, mode=up_mode),
                                          spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)))
                # nn.ConvTranspose2d(in_channel, out_channel, kernel_size=1, stride=2, padding=0,
                #                   output_padding=1)  # ？
            else:
                self.c_sc = spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0))

    def shortcut(self, x):
        if self.learnable_sc:
            return self.c_sc(x)
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)



class GBlockEm(nn.Module):  # 可以输入条件类标的，带有条件BN层的reblock
    def __init__(self, in_channel, out_channel, hidden_channel=None, ksize=(3, 3), pad=1, upsample=False, num_em=0,up_mode='bilinear'):
        super(GBlockEm, self).__init__()
        self.upsample = upsample
        hidden_channel = out_channel if hidden_channel is None else hidden_channel
        self.learnable_sc = in_channel != out_channel or upsample
        self.num_embed = num_em
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        if self.upsample:
            self.c1 = nn.Sequential(nn.Upsample(scale_factor=2, mode=up_mode),
                                    nn.Conv2d(in_channel, out_channel, kernel_size=ksize, padding=pad))
        else:
            self.c1 = nn.Conv2d(in_channel, hidden_channel, kernel_size=ksize, padding=pad)
        # self.c1=nn.Conv2d(in_channel,hidden_channel,kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channel, out_channel, kernel_size=ksize, padding=pad)
        if num_em > 0:
            self.b1 = CategoricalConditionalBatchNorm2dEmbed(num_em, in_channel)
            self.b2 = CategoricalConditionalBatchNorm2dEmbed(num_em, hidden_channel)
        else:
            self.b1 = nn.BatchNorm2d(in_channel)
            self.b2 = nn.BatchNorm2d(hidden_channel)

        if self.learnable_sc:
            if self.upsample:
                self.c_sc = nn.Sequential(nn.Upsample(scale_factor=2, mode=up_mode),
                                          nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0))
            else:
                self.c_sc = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)

    def residual(self, x, y=None, **kwargs):
        h = x
        h = self.b1(h, y, **kwargs) if y is not None else self.b1(h, **kwargs)
        h = self.act1(h)
        h = self.c1(h)
        h = self.b2(h, y, **kwargs) if y is not None else self.b2(h, **kwargs)
        h = self.act2(h)
        h = self.c2(h)
        return h

    def shotcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x, y, **kwargs):
        return self.residual(x, y, **kwargs) + self.shotcut(x)

class SNGBlockEm(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=None, ksize=(3, 3), pad=1, upsample=False, num_em=0,up_mode='bilinear'):
        super(SNGBlockEm, self).__init__()
        self.upsample = upsample
        hidden_channel = out_channel if hidden_channel is None else hidden_channel
        self.learnable_sc = in_channel != out_channel or upsample
        self.num_embed = num_em
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        if self.upsample:
            self.c1 = nn.Sequential(nn.Upsample(scale_factor=2, mode=up_mode),
                                    spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=ksize, padding=pad)))
        else:
            self.c1 = spectral_norm(nn.Conv2d(in_channel, hidden_channel, kernel_size=ksize, padding=pad))
        # self.c1=nn.Conv2d(in_channel,hidden_channel,kernel_size=ksize, padding=pad)
        self.c2 = spectral_norm(nn.Conv2d(hidden_channel, out_channel, kernel_size=ksize, padding=pad))
        if num_em > 0:
            self.b1 = SNCategoricalConditionalBatchNorm2dEmbed(num_em, in_channel)
            self.b2 = SNCategoricalConditionalBatchNorm2dEmbed(num_em, hidden_channel)
        else:
            self.b1 = nn.BatchNorm2d(in_channel)
            self.b2 = nn.BatchNorm2d(hidden_channel)

        if self.learnable_sc:
            if self.upsample:
                self.c_sc = nn.Sequential(nn.Upsample(scale_factor=2, mode=up_mode),
                                          spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)))
            else:
                self.c_sc = spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0))

    def residual(self, x, y=None, **kwargs):
        h = x
        h = self.b1(h, y, **kwargs) if y is not None else self.b1(h, **kwargs)
        h = self.act1(h)
        h = self.c1(h)
        h = self.b2(h, y, **kwargs) if y is not None else self.b2(h, **kwargs)
        h = self.act2(h)
        h = self.c2(h)
        return h

    def shotcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x, y, **kwargs):
        return self.residual(x, y, **kwargs) + self.shotcut(x)



class GBlockAttr(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=None, ksize=(3, 3), pad=1, upsample=False, num_attr=0,up_mode='bilinear'):
        super(GBlockAttr, self).__init__()
        self.upsample = upsample
        hidden_channel = out_channel if hidden_channel is None else hidden_channel
        self.learnable_sc = in_channel != out_channel or upsample
        self.num_attr = num_attr
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        if self.upsample:
            self.c1 = nn.Sequential(nn.Upsample(scale_factor=2, mode=up_mode),
                                    nn.Conv2d(in_channel, out_channel, kernel_size=ksize, padding=pad))
        else:
            self.c1 = nn.Conv2d(in_channel, hidden_channel, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channel, out_channel, kernel_size=ksize, padding=pad)

        if num_attr > 0:
            self.b1 = CategoricalConditionalBatchNorm2dAttr(num_attr, in_channel)
            self.b2 = CategoricalConditionalBatchNorm2dAttr(num_attr, hidden_channel)
        else:
            self.b1 = nn.BatchNorm2d(in_channel)
            self.b2 = nn.BatchNorm2d(hidden_channel)

        if self.learnable_sc:
            if self.upsample:
                self.c_sc = nn.Sequential(nn.Upsample(scale_factor=2, mode=up_mode),
                                          nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), padding=(0, 0)))
            else:
                self.c_sc = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), padding=(0, 0))

    def residual(self, x, y=None, **kwargs):
        h = x
        #h = self.b1(h, y, **kwargs) if y is not None else self.b1(h, **kwargs)
        h=self.b1(h,y)
        h = self.act1(h)
        h = self.c1(h)
        #h = self.b2(h, y, **kwargs) if y is not None else self.b2(h, **kwargs)
        h=self.b2(h,y)
        h = self.act2(h)
        h = self.c2(h)
        return h

    def shotcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x, y, **kwargs):
        return self.residual(x, y, **kwargs) + self.shotcut(x)

class SNGBlockAttr(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=None, ksize=(3, 3), pad=1, upsample=False, num_attr=0,up_mode='bilinear'):
        super(SNGBlockAttr, self).__init__()
        self.upsample = upsample
        hidden_channel = out_channel if hidden_channel is None else hidden_channel
        self.learnable_sc = in_channel != out_channel or upsample
        self.num_attr = num_attr
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        if self.upsample:
            self.c1 = nn.Sequential(nn.Upsample(scale_factor=2, mode=up_mode),
                                    spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=ksize, padding=pad)))
        else:
            self.c1 = spectral_norm(nn.Conv2d(in_channel, hidden_channel, kernel_size=ksize, padding=pad))
        self.c2 = spectral_norm(nn.Conv2d(hidden_channel, out_channel, kernel_size=ksize, padding=pad))

        if num_attr > 0:
            self.b1 = SNCategoricalConditionalBatchNorm2dAttr(num_attr, in_channel)
            self.b2 = SNCategoricalConditionalBatchNorm2dAttr(num_attr, hidden_channel)
        else:
            self.b1 = nn.BatchNorm2d(in_channel)
            self.b2 = nn.BatchNorm2d(hidden_channel)

        if self.learnable_sc:
            if self.upsample:
                self.c_sc = nn.Sequential(nn.Upsample(scale_factor=2, mode=up_mode),
                                          spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), padding=(0, 0))))
            else:
                self.c_sc = spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), padding=(0, 0)))

    def residual(self, x, y=None, **kwargs):
        h = x
        # h = self.b1(h, y, **kwargs) if y is not None else self.b1(h, **kwargs)
        h = self.b1(h, y)
        h = self.act1(h)
        h = self.c1(h)
        # h = self.b2(h, y, **kwargs) if y is not None else self.b2(h, **kwargs)
        h = self.b2(h, y)
        h = self.act2(h)
        h = self.c2(h)
        return h

    def shotcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x, y, **kwargs):
        return self.residual(x, y, **kwargs) + self.shotcut(x)



class GBlockAttrDeconv(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=None, ksize=(3, 3), pad=1, upsample=False, num_attr=0,up_mode='bilinear'):
        super(GBlockAttrDeconv, self).__init__()
        self.upsample = upsample
        hidden_channel = out_channel if hidden_channel is None else hidden_channel
        self.learnable_sc = in_channel != out_channel or upsample
        self.num_attr = num_attr
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        if self.upsample:
            self.c1 = nn.ConvTranspose2d(in_channel, hidden_channel, kernel_size=ksize, stride=2,
                                         padding=pad,
                                         output_padding=1)
            # nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
            # nn.Conv2d(in_channel, out_channel, kernel_size=ksize, padding=pad))
        else:
            self.c1 = nn.Conv2d(in_channel, hidden_channel, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channel, out_channel, kernel_size=ksize, padding=pad)

        if num_attr > 0:
            self.b1 = CategoricalConditionalBatchNorm2dAttr(num_attr, in_channel)
            self.b2 = CategoricalConditionalBatchNorm2dAttr(num_attr, hidden_channel)
        else:
            self.b1 = nn.BatchNorm2d(in_channel)
            self.b2 = nn.BatchNorm2d(hidden_channel)

        if self.learnable_sc:
            if self.upsample:
                self.c_sc = nn.Sequential(nn.Upsample(scale_factor=2, mode=up_mode),
                                          nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), padding=(0, 0)))
            else:
                self.c_sc = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), padding=(0, 0))

    def residual(self, x, y=None, **kwargs):
        h = x
        h = self.b1(h, y, **kwargs) if y is not None else self.b1(h, **kwargs)
        h = self.act1(h)
        h = self.c1(h)
        h = self.b2(h, y, **kwargs) if y is not None else self.b2(h, **kwargs)
        h = self.act2(h)
        h = self.c2(h)
        return h

    def shotcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x, y, **kwargs):
        return self.residual(x, y, **kwargs) + self.shotcut(x)


class SNDBlock(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=None, ksize=3, pad=1, downsample=False):
        super(SNDBlock, self).__init__()
        self.downsample = downsample
        self.skip_connection = (in_channel != out_channel) or downsample
        hidden_channel = in_channel if hidden_channel is None else hidden_channel
        self.residual = nn.Sequential(nn.ReLU(inplace=True),
                                      spectral_norm(
                                          nn.Conv2d(in_channel, hidden_channel, kernel_size=ksize, padding=pad)),
                                      nn.ReLU(inplace=True),
                                      spectral_norm(
                                          nn.Conv2d(hidden_channel, out_channel, kernel_size=ksize, padding=pad)),
                                      )

        # self.c1 = spectral_norm(nn.Conv2d(in_channel, hidden_channel, kernel_size=ksize, padding=pad))
        # SNConv2d(in_channel, hidden_channel, kernel_size=ksize, padding=pad)
        # self.c2 = spectral_norm(nn.Conv2d(hidden_channel, out_channel, kernel_size=ksize, padding=pad))
        # SNConv2d(hidden_channel, out_channel, kernel_size=ksize, padding=pad)
        self.act_func = nn.ReLU(inplace=True)
        if self.skip_connection:
            self.c_sc = spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0))
            # SNConv2d(in_channel, out_channel, kernel_size=1, padding=0)  #
        if downsample:
            self._downsample = nn.AvgPool2d(2)

    def residual_down(self, x):
        h = self.residual(x)
        if self.downsample:
            h = self._downsample(h)
        return h

    def shortcut(self, x):
        if self.skip_connection:
            x = self.c_sc(x)
            if self.downsample:
                return self._downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual_down(x) + self.shortcut(x)


class SNDOptimizedBlock(nn.Module):
    def __init__(self, in_channel, out_channel, ksize=3, pad=1):
        super(SNDOptimizedBlock, self).__init__()
        self.residual = nn.Sequential(spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=ksize, padding=pad)),
                                      nn.ReLU(inplace=True),
                                      spectral_norm(
                                          nn.Conv2d(out_channel, out_channel, kernel_size=ksize, padding=pad)),
                                      nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(nn.AvgPool2d(2),
                                      spectral_norm(nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self,in_channel):
        super(AttentionBlock, self).__init__()
        self.in_channel=in_channel
        self.query_conv=nn.Conv2d(in_channel,out_channels=(in_channel//8),kernel_size=1)
        self.key_conv=nn.Conv2d(in_channel,out_channels=(in_channel//8),kernel_size=1)
        self.value_conv=nn.Conv2d(in_channel,in_channel,kernel_size=1)
        self.gamma=nn.parameter.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        """
                    inputs :
                        x : input feature maps( B X C X W X H)
                    returns :
                        out : self attention value + input feature
                        attention: B X N X N (N is Width*Height)

                        thanks to https://github.com/heykeetae/Self-Attention-GAN/blob/8714a54ba5027d680190791ba3a6bb08f9c9a129/sagan_models.py#L8
                """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out#, attention




class SNAttentionBlock(nn.Module):
    def __init__(self,in_channel):
        super(SNAttentionBlock, self).__init__()
        self.in_channel = in_channel
        self.query_conv = spectral_norm(nn.Conv2d(in_channel, out_channels=(in_channel // 8), kernel_size=1))
        self.key_conv = spectral_norm(nn.Conv2d(in_channel, out_channels=(in_channel // 8), kernel_size=1))
        self.value_conv = spectral_norm(nn.Conv2d(in_channel, in_channel, kernel_size=1))
        self.gamma = nn.parameter.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x):
        """
                    inputs :
                        x : input feature maps( B X C X W X H)
                    returns :
                        out : self attention value + input feature
                        attention: B X N X N (N is Width*Height)

                        thanks to https://github.com/heykeetae/Self-Attention-GAN/blob/8714a54ba5027d680190791ba3a6bb08f9c9a129/sagan_models.py#L8
                """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out#, attention

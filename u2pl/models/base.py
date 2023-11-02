import torch
import torch.nn as nn
from torch.nn import functional as F
from tricks.attention import Cbam, CoordAtt, DANetAtt
#from mmcv.cnn import ConvModule
import torchsnooper
import warnings

def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def get_syncbn():
    # return nn.BatchNorm2d
    return nn.SyncBatchNorm


class ASPP(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(
        self, in_planes, inner_planes=256, sync_bn=False, dilations=(12, 24, 36)  # in_planes：输入通道数；inner_planes：输出通道数；dilations：膨胀率
    ):
        super(ASPP, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d        # norm_layer = nn.SyncBatchNorm
        self.conv1 = nn.Sequential(                # conv1：用一个尺寸为input大小的池化层将input池化为1×1，再用一个1×1的卷积进行降维，最后上采样回原始输入大小
            nn.AdaptiveAvgPool2d((1, 1)),          # AdaptiveAvgPool2d：自适应均值池化，可以将将各通道的特征图分别压缩至1×1，从而提取各通道的特征，进而获取全局的特征
            nn.Conv2d(                             # 利用1*1卷积对上一步获取的特征进行进一步的提取，并降维
                in_planes,
                inner_planes,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(                 # conv2：用一个1×1的卷积对input进行降维
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(                 # conv3：用一个padding为12，dilation为12，核大小为3×3的卷积层进行卷积
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[0],
                dilation=dilations[0],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(                 # conv4：用一个padding为24，dilation为24，核大小为3×3的卷积层进行卷积
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[1],
                dilation=dilations[1],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(                 # conv5：用一个padding为36，dilation为36，核大小为3×3的卷积层进行卷积
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[2],
                dilation=dilations[2],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )

        self.out_planes = (len(dilations) + 2) * inner_planes    # out_planes = 5*inner_planes

    def get_outplanes(self):
        return self.out_planes

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=True     # 采用双线性插值方法将conv1的输出上采样成形状（h, w）,使它和其他conv层输出形状匹配
        )
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        aspp_out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)            # 最后将这五层的输出在dim=1维进行concat得到最终输出
        return aspp_out

# @torchsnooper.snoop() #把这个声明放在想要输出的函数前
class attention_ASPP(nn.Module):
    def __init__(
            self, in_planes, inner_planes=256, sync_bn=False, dilations=(3, 6, 12, 24, 36)
            # in_planes：输入通道数；inner_planes：输出通道数；dilations：膨胀率
    ):
        super(attention_ASPP, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d  # norm_layer = nn.SyncBatchNorm
        ### 添加注意力机制
        self.corrdatt    =  CoordAtt(inner_planes, inner_planes)

        self.conv1 = nn.Sequential(  # conv1：用一个1×1的卷积对input进行降维
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(  # conv3：用一个padding为12，dilation为12，核大小为3×3的卷积层进行卷积
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[0],
                dilation=dilations[0],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(  # conv4：用一个padding为24，dilation为24，核大小为3×3的卷积层进行卷积
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[1],
                dilation=dilations[1],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(  # conv5：用一个padding为36，dilation为36，核大小为3×3的卷积层进行卷积
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[2],
                dilation=dilations[2],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(  # conv5：用一个padding为36，dilation为36，核大小为3×3的卷积层进行卷积
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[3],
                dilation=dilations[3],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )

        self.conv6 = nn.Sequential(  # conv5：用一个padding为36，dilation为36，核大小为3×3的卷积层进行卷积
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[4],
                dilation=dilations[4],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )

        self.out_planes = len(dilations) * inner_planes  # out_planes = 5*inner_planes

    def get_outplanes(self):
        return self.out_planes

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = self.conv1(x)
        feat1 = self.corrdatt(feat1)

        feat2 = self.conv2(x)
        feat2 = self.corrdatt(feat2)
        feat2 += feat1

        feat3 = self.conv3(x)
        feat3 = self.corrdatt(feat3)
        feat3 += feat1

        feat4 = self.conv4(x)
        feat4 = self.corrdatt(feat4)
        feat4 += feat1

        feat5 = self.conv5(x)
        feat5 = self.corrdatt(feat5)
        feat5 += feat1

        feat6 = self.conv6(x)
        feat6 = self.corrdatt(feat6)
        feat6 += feat1

        aspp_out = torch.cat((feat2, feat3, feat4, feat5, feat6), 1)  # 最后将这五层的输出在dim=1维进行concat得到最终输出
        return aspp_out

class DAnet(nn.Module):
    def __init__(self, in_planes, inner_planes=1280, sync_bn=False):
        super(DAnet, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.da_att = DANetAtt(in_planes, inner_planes)
        self.out_planes = inner_planes                  # 不直接写1280而是写成inner_planes * 5是为了方便decoder的编写

    def get_outplanes(self):
        return self.out_planes

    def forward(self, x):
        c, n, h, w = x.size()
        feat = self.da_att(x)
        DAnet_out = feat[0]
        return DAnet_out

# @torchsnooper.snoop() #把这个声明放在想要输出的函数前

class LHFM(nn.Module):
    '''
    效果最好的方法
    '''
    def __init__(self, in_planes=512, ratio=16):   # in_planes：输入通道数 原本是512；inner_planes：输出通道数
        super(LHFM5, self).__init__()

        # Concat之后经过的3*3Conv + BatchNorm + Relu，是通道维度变为256
        # self.post_conv = nn.Sequential(
        #     nn.Conv2d(in_planes, inner_planes, kernel_size=3, bias=False,
        #               dilation=1, stride=1, padding=1),
        #     nn.BatchNorm2d(inner_planes),
        #     nn.ReLU()
        # )

        ### 空间注意力分支(SAB)，输出为128*128*1
        # self.conv1 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False),
        #                         nn.Conv2d(inner_planes, inner_planes, kernel_size=3, padding=1, groups=inner_planes),  # 沿深度方向的3*3分组卷积，可以生成256个128*128*1的feature map，然后再把他们Concat起来
        #                         nn.Sigmoid())

        ### 通道注意力分支(CAB)，输出为1*1*256
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(  # 利用1*1卷积对上一步获取的特征进行进一步的提取，并降维
                in_planes,
                in_planes // ratio,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_planes // ratio,
                in_planes,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x1, x2):  # x1:低级特征 x2:高级特征
        # concat方法 和LFHM2的区别
        y = torch.cat((x1, x2), 1)   #y:([2, 512, 129, 129])
        ### 通道注意力(SE)
        avg_out2 = self.avg_pool(y)
        avg_out2 = self.conv2(avg_out2)
        feat2 = avg_out2
        feat2 = self.sigmoid(feat2)
        # print(feat2.shape)    #   feat2：([2, 256, 1, 1])   通道注意力图
        y_out = y * feat2
        LHFM5_out = y_out
        return LHFM5_out

class PPM(nn.Module):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners, **kwargs):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        **kwargs)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

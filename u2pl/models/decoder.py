import torch
import torch.nn as nn
from torch.nn import functional as F
from tricks.attention import Cbam, Eca_Model, CoordAtt, DANetAtt
import torchsnooper
#from mmcv.cnn import ConvModule
from .base import ASPP, get_syncbn, attention_ASPP, DAnet, attention_ASPP_test1, LHFM, LHFM2, LHFM3, LHFM4, LHFM5, PPM


class dec_deeplabv3(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes=19,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 24, 36),
    ):
        super(dec_deeplabv3, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d

        self.aspp = ASPP(
            in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations
        )
        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                256,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        aspp_out = self.aspp(x)
        res = self.head(aspp_out)
        return res


class dec_deeplabv3_plus(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes=6,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 24, 36),
        rep_head=True,
    ):
        super(dec_deeplabv3_plus, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d                         # norm_layer = nn.SyncBatchNorm
        self.rep_head = rep_head                                                         # True

        self.low_conv = nn.Sequential(                                                   # 分支一，提取低级特征
            nn.Conv2d(256, 256, kernel_size=1),
            norm_layer(256),
            nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(                                                                # 分支二，提取高级特征
            in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations
        )

        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                256,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.classifier = nn.Sequential(                                                # 分割头的输出，用于有监督Ls和无监督的Lu的训练
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True), # 输出通道为类别数，即语义结果的one-hot表示
        )

        if self.rep_head:                                                               # 表示头的输出，用于对比损失Lc的训练

            self.representation = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),     # 输出结果是256维的表示空间
            )

    def forward(self, x):
        x1, x2, x3, x4 = x                          # x是ResNet的输出，它分别返回4个残差块的输出
        aspp_out = self.aspp(x4)                    # 将最后一个残差块的输出输入到ASPP结构中，返回ASPP结果，即将五个输出cat到一起
        # print('aspp_out shape:', aspp_out.shape)
        low_feat = self.low_conv(x1)                # 第一个残差块的输出输入到low_conv中，得到low_feat即低级特征
        aspp_out = self.head(aspp_out)              # 将ASPP的输出输入到head函数里，得到高级特征
        h, w = low_feat.size()[-2:]                 # h，w为low_feat（低级特征）的长和宽
        aspp_out = F.interpolate(                   # 将高级特征进行二线性上采样到和低级特征一样的形状
            aspp_out, size=(h, w), mode="bilinear", align_corners=True
        )
        aspp_out = torch.cat((low_feat, aspp_out), dim=1)   # 将高级特征和低级特征cat到一起
        # print('aspp_out shape:', aspp_out.shape)            # (2, 512, 128, 128)
        res = {"pred": self.classifier(aspp_out)}           # 得到分割头的输出

        if self.rep_head:
            res["rep"] = self.representation(aspp_out)      # 得到表示头的输出

        return res                                          # res是一个列表，有两个元素，分别为‘pred’和‘rep’

class PSPHead(nn.Module):
    """Pyramid Scene Parsing Network.

    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6)):
        super(PSPHead, self).__init__()
        assert isinstance(pool_scales, (list, tuple))
        self.pool_scales = pool_scales
        self.psp_modules = PPM(
            self.pool_scales,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        feats = self.bottleneck(psp_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output

class Aux_Module(nn.Module):
    def __init__(self, in_planes, num_classes=19, sync_bn=False):
        super(Aux_Module, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.aux = nn.Sequential(
            nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        res = self.aux(x)
        return res

# @torchsnooper.snoop() #把这个声明放在想要输出的函数前

class SCDMNet(nn.Module):
    # 将高级特征和低级特征的融合由concat变为LFHM5
    def __init__(
        self,
        in_planes,
        num_classes=6,
        inner_planes=256,
        out_planes=1280,
        sync_bn=False,
        dilations=(3, 6, 12, 24, 36),
        rep_head=True,
    ):
        super(SCDMNet, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.rep_head = rep_head

        self.low_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            norm_layer(256),
            nn.ReLU(inplace=True)
        )

        self.aspp = attention_ASPP(
            in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations
        )

        self.danet = DAnet(in_planes, inner_planes=out_planes, sync_bn=sync_bn)

        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes() + self.danet.get_outplanes(),
                256,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

        if self.rep_head:
            self.representation = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            )

        ### 低级高级特征融合模块
        self.LHFM5 = LHFM5(in_planes=512)

    def forward(self, x):
        x1, x2, x3, x4 = x
        # x4 = self.corrdatt1(x4)
        aspp_out = self.aspp(x4)
        # print('aspp_out shape:', aspp_out.shape)
        danet_out = self.danet(x4)
        higt_out  = torch.cat((aspp_out, danet_out), dim=1)
        # x1 = self.corrdatt2(x1)
        low_feat = self.low_conv(x1)
        higt_out = self.head(higt_out)
        h, w = low_feat.size()[-2:]
        higt_out = F.interpolate(
            higt_out, size=(h, w), mode="bilinear", align_corners=True
        )
        aspp_out = self.LHFM5(low_feat, higt_out)

        res = {"pred": self.classifier(aspp_out)}

        if self.rep_head:
            res["rep"] = self.representation(aspp_out)

        return res

class LHFM_deeplabv3_plus(nn.Module):
    # U2PL+LHFM
    def __init__(
        self,
        in_planes,
        num_classes=6,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 24, 36),
        rep_head=True,
    ):
        super(LHFM_deeplabv3_plus, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.rep_head = rep_head

        self.low_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            norm_layer(256),
            nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(
            in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations
        )


        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                256,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

        if self.rep_head:
            self.representation = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            )


        self.LHFM5 = LHFM5(in_planes=512)

    def forward(self, x):
        x1, x2, x3, x4 = x
        aspp_out = self.aspp(x4)
        low_feat = self.low_conv(x1)
        higt_out = self.head(aspp_out)
        h, w = low_feat.size()[-2:]
        higt_out = F.interpolate(
            higt_out, size=(h, w), mode="bilinear", align_corners=True
        )
        aspp_out = self.LHFM5(low_feat, higt_out)

        res = {"pred": self.classifier(aspp_out)}

        if self.rep_head:
            res["rep"] = self.representation(aspp_out)

        return res
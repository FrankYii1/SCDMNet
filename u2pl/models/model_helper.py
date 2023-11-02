import importlib
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from .decoder import Aux_Module


class ModelBuilder(nn.Module):
    def __init__(self, net_cfg):
        super(ModelBuilder, self).__init__()
        self._sync_bn = net_cfg["sync_bn"]                                              # True
        self._num_classes = net_cfg["num_classes"]                                      # num_classes = 6

        self.encoder = self._build_encoder(net_cfg["encoder"])                          # encoder = u2pl.models.resnet.resnet101
        self.decoder = self._build_decoder(net_cfg["decoder"])                          # decoder = u2pl.models.decoder.dec_deeplabv3_plus


        self._use_auxloss = True if net_cfg.get("aux_loss", False) else False           # False
        self.fpn = True if net_cfg["encoder"]["kwargs"].get("fpn", False) else False    # True
        if self._use_auxloss:                                                           # False
            cfg_aux = net_cfg["aux_loss"]
            self.loss_weight = cfg_aux["loss_weight"]
            self.auxor = Aux_Module(
                cfg_aux["aux_plane"], self._num_classes, self._sync_bn
            )

    def _build_encoder(self, enc_cfg):
        enc_cfg["kwargs"].update({"sync_bn": self._sync_bn})
        encoder = self._build_module(enc_cfg["type"], enc_cfg["kwargs"])
        return encoder

    def _build_decoder(self, dec_cfg):
        dec_cfg["kwargs"].update(
            {
                "in_planes": self.encoder.get_outplanes(),          # 128
                "sync_bn": self._sync_bn,                           # True
                "num_classes": self._num_classes,                   # num_classes = 6
            }
        )
        decoder = self._build_module(dec_cfg["type"], dec_cfg["kwargs"])
        return decoder

    def _build_module(self, mtype, kwargs):
        module_name, class_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    def forward(self, x):
        if self._use_auxloss:
            if self.fpn:
                # feat1 used as dsn loss as default, f1 is layer2's output as default
                f1, f2, feat1, feat2 = self.encoder(x)
                outs = self.decoder([f1, f2, feat1, feat2])
            else:
                feat1, feat2 = self.encoder(x)
                outs = self.decoder(feat2)

            pred_aux = self.auxor(feat1)

            outs.update({"aux": pred_aux})
            return outs
        else:
            feat = self.encoder(x)
            print('Resnet outshape:', np.array(feat).shape)
            outs = self.decoder(feat)
            return outs

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from geoseg.models.pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from geoseg.models.decoders import FQPD


class PFQGNet(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='pvt_v2_b2',
                 pretrained=True,
                 window_size=8,
                 num_classes=6,
                 expansion_factor=4,
                 dw_parallel=True  #
                 ):
        super().__init__()
        if backbone_name == 'pvt_v2_b2':
            self.backbone = pvt_v2_b2()
            encoder_channels = [64, 128, 320, 512]
            if pretrained:
                #path = './pretrain_weights/pvt/pvt_v2_b2.pth'
                path = 'E:\code\GeoSeg-main\pretrain_weights\pvt\pvt_v2_b2.pth'
                state_dict = torch.load(path, map_location='cpu')
                model_dict = self.backbone.state_dict()
                matched_weights = {k: v for k, v in state_dict.items() if k in model_dict}
                model_dict.update(matched_weights)
                self.backbone.load_state_dict(model_dict)
                print(f"[✓] Loaded local PVTv2 weights from: {path}")
        else:
            if 'resnet' in backbone_name:
                self.backbone = timm.create_model(
                    backbone_name,
                    features_only=True,
                    output_stride=32,
                    out_indices=(1, 2, 3, 4),
                    pretrained=pretrained
                )
            else:
                self.backbone = timm.create_model(
                    backbone_name,
                    features_only=True,
                    out_indices=(1, 2, 3, 4),
                    pretrained=pretrained
                )

        encoder_channels = [64, 128, 320, 512]
        channels = [
            encoder_channels[-1],  # res4
            encoder_channels[-2],  # res3
            encoder_channels[-3],  # res2
            decode_channels
        ]

        self.decoder = FQPD(
            channels=channels,
            kernel_sizes=[1, 3, 5],
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel,
            add=True,
            lgag_ks=3,
            activation='relu6'
        )

        self.seg_head = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels, kernel_size=3),
            nn.Dropout2d(dropout),
            nn.Conv2d(decode_channels, num_classes, kernel_size=1)
        )

        if self.training:
            self.aux_head = nn.Sequential(
                ConvBNReLU(channels[2], decode_channels, kernel_size=3),
                nn.Conv2d(decode_channels, num_classes, kernel_size=1)

            )
    def forward(self, x):
        h, w = x.size()[-2:]

        res1, res2, res3, res4 = self.backbone(x)

        dec_features = self.decoder(res4, [res3, res2, res1])
        d1 = dec_features[-1]

        out = self.seg_head(d1)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        if self.training:
            aux = self.aux_head(dec_features[-2])

            aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=False)

            return out, aux
        return out
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True)
        )



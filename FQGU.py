import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import warnings
import numpy as np

def create_hamming_window(M, N):

    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    hamming_2d = np.outer(hamming_x, hamming_y)
    return hamming_2d

def interpolate_feature(input,
                        size=None,
                        scale_factor=None,
                        mode='nearest',
                        align_corners=None,
                        warning=True):

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

try:
    from mmcv.ops.carafe import normal_init as init_normal, xavier_init as init_xavier, carafe as carafe_op
except ImportError:

    def init_xavier(module: nn.Module,
                    gain: float = 1,
                    bias: float = 0,
                    distribution: str = 'normal') -> None:

        assert distribution in ['uniform', 'normal']
        if hasattr(module, 'weight') and module.weight is not None:
            if distribution == 'uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            else:
                nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def carafe_op(x, normed_mask, kernel_size, group=1, up=1):

        b, c, h, w = x.shape
        _, m_c, m_h, m_w = normed_mask.shape
        assert m_h == up * h
        assert m_w == up * w
        pad = kernel_size // 2
        pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
        unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
        unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)
        unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='nearest')
        unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, m_h, m_w)
        normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
        res = (unfold_x * normed_mask).sum(dim=2).reshape(b, c, m_h, m_w)
        return res

    def init_normal(module, mean=0, std=1, bias=0):

        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)


def init_constant(module, val, bias=0):

    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class FQGU(nn.Module):

    def __init__(self,
                 hr_channels,
                 lr_channels,
                 scale_factor=1,
                 lfqg_kernel=5,
                 hfqg_kernel=3,
                 up_group=1,
                 encoder_kernel=3,
                 encoder_dilation=1,
                 compressed_channels=96,
                 align_corners=False,
                 upsample_mode='nearest',
                 comp_feat_upsample=True,
                 use_hfqg=True,
                 use_lfqg=True,
                 hr_residual=True,
                 semi_conv=True,
                 hamming_window=True,
                 **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        self.lfqg_kernel = lfqg_kernel
        self.hfqg_kernel = hfqg_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)
        self.content_encoder = nn.Conv2d(
            self.compressed_channels,
            lfqg_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)

        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.hr_residual = hr_residual
        self.use_hfqg = use_hfqg
        self.use_lfqg = use_lfqg
        self.semi_conv = semi_conv
        self.comp_feat_upsample = comp_feat_upsample

        self.proj_lr = nn.Conv2d(lr_channels, compressed_channels, kernel_size=1)

        if self.use_hfqg:
            self.content_encoder2 = nn.Conv2d(
                self.compressed_channels,
                hfqg_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
        self.hamming_window = hamming_window
        lfqg_pad = 0
        hfqg_pad = 0
        if self.hamming_window:
            self.register_buffer('hamming_lfqg', torch.FloatTensor(
                create_hamming_window(lfqg_kernel + 2 * lfqg_pad, lfqg_kernel + 2 * lfqg_pad))[None, None,])
            self.register_buffer('hamming_hfqg', torch.FloatTensor(
                create_hamming_window(hfqg_kernel + 2 * hfqg_pad, hfqg_kernel + 2 * hfqg_pad))[None, None,])
        else:
            self.register_buffer('hamming_lfqg', torch.FloatTensor([1.0]))
            self.register_buffer('hamming_hfqg', torch.FloatTensor([1.0]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_xavier(m, distribution='uniform')
        init_normal(self.content_encoder, std=0.001)
        if self.use_hfqg:
            init_normal(self.content_encoder2, std=0.001)

    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel ** 2))

        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask

    def forward(self, hr_feat, lr_feat, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self._forward, hr_feat, lr_feat)
        else:
            return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)
        if self.semi_conv:
            if self.comp_feat_upsample:
                if self.use_hfqg:
                    mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat)
                    mask_hr_init = self.kernel_normalizer(mask_hr_hr_feat, self.hfqg_kernel,
                                                          hamming=self.hamming_hfqg)
                    compressed_hr_feat = compressed_hr_feat + compressed_hr_feat - carafe_op(compressed_hr_feat,
                                                                                             mask_hr_init,
                                                                                             self.hfqg_kernel,
                                                                                             self.up_group,
                                                                                             1)
                    mask_lr_hr_feat = self.content_encoder(compressed_hr_feat)
                    mask_lr_init = self.kernel_normalizer(mask_lr_hr_feat, self.lfqg_kernel,
                                                          hamming=self.hamming_lfqg)
                    mask_lr_lr_feat_lr = self.content_encoder(compressed_lr_feat)
                    mask_lr_lr_feat = F.interpolate(
                        carafe_op(mask_lr_lr_feat_lr, mask_lr_init, self.lfqg_kernel, self.up_group, 2),
                        size=compressed_hr_feat.shape[-2:], mode='nearest')
                    mask_lr = mask_lr_hr_feat + mask_lr_lr_feat

                    mask_lr_init = self.kernel_normalizer(mask_lr, self.lfqg_kernel,
                                                          hamming=self.hamming_lfqg)
                    mask_hr_lr_feat = F.interpolate(
                        carafe_op(self.content_encoder2(compressed_lr_feat), mask_lr_init, self.lfqg_kernel,
                                  self.up_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest')
                    mask_hr = mask_hr_hr_feat + mask_hr_lr_feat
                else:
                    lr_feat = F.interpolate(lr_feat, size=hr_feat.shape[-2:], mode='bilinear', align_corners=False)
                    if lr_feat.shape[1] != hr_feat.shape[1]:
                        self.align_channels = nn.Conv2d(lr_feat.shape[1], hr_feat.shape[1], kernel_size=1).to(
                            lr_feat.device)
                        lr_feat = self.align_channels(lr_feat)
                    out = hr_feat + lr_feat
                    return hr_feat, lr_feat, out
            else:
                mask_lr = self.content_encoder(compressed_hr_feat) + F.interpolate(
                    self.content_encoder(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode='nearest')
                if self.use_hfqg:
                    mask_hr = self.content_encoder2(compressed_hr_feat) + F.interpolate(
                        self.content_encoder2(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode='nearest')
        else:
            compressed_x = F.interpolate(compressed_lr_feat, size=compressed_hr_feat.shape[-2:],
                                         mode='nearest') + compressed_hr_feat
            mask_lr = self.content_encoder(compressed_x)
            if self.use_hfqg:
                mask_hr = self.content_encoder2(compressed_x)

        mask_lr = self.kernel_normalizer(mask_lr, self.lfqg_kernel, hamming=self.hamming_lfqg)
        if self.semi_conv:
            lr_feat = carafe_op(lr_feat, mask_lr, self.lfqg_kernel, self.up_group, 2)
        else:
            lr_feat = interpolate_feature(
                input=lr_feat,
                size=hr_feat.shape[2:],
                mode=self.upsample_mode,
                align_corners=None if self.upsample_mode == 'nearest' else self.align_corners)
            lr_feat = carafe_op(lr_feat, mask_lr, self.lfqg_kernel, self.up_group, 1)

        if self.use_hfqg:
            mask_hr = self.kernel_normalizer(mask_hr, self.hfqg_kernel, hamming=self.hamming_hfqg)
            hr_feat_hf = hr_feat - carafe_op(hr_feat, mask_hr, self.hfqg_kernel, self.up_group, 1)
            if self.hr_residual:
                hr_feat = hr_feat_hf + hr_feat
            else:
                hr_feat = hr_feat_hf

        lr_feat = self.proj_lr(lr_feat)
        return mask_lr, hr_feat, lr_feat
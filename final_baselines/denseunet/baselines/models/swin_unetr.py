"""
Swin UNETR for MRI → tau-PET regression.

Source: Hatamizadeh et al., "Swin UNETR: Swin Transformers for Semantic Segmentation of
Brain Tumors in MRI Images" (arXiv:2201.01266), implementation = MONAI's
monai.networks.nets.SwinUNETR (the authors' own release).

Configuration (paper Table 1): patch size 2, embedding C = 48 (feature_size), depths
[2,2,2,2], heads [3,6,12,24], window 7, instance-norm residual-block decoder; 62.2 M
parameters here (61.98 M in the paper with 4 input channels).

Deviations for this task (documented in docs/BASELINES.md):
  * Regression, not segmentation: the 1×1×1 output conv is kept linear (no sigmoid) and
    the model is trained with the same DK86-masked MSE as the Dense-U-Net baseline, so the
    two deterministic regressors differ only in architecture.
  * Swin UNETR needs spatial dims divisible by 32; the lab volume (96,112,96) is padded
    to (96,128,96) inside forward() and the output cropped back.
"""
import torch.nn as nn

from .common import crop_pads, pad_to_multiple


class SwinUNETRRegressor(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, feature_size=48, depths=(2, 2, 2, 2),
                 num_heads=(3, 6, 12, 24), window_size=7, drop_path_rate=0.0,
                 use_checkpoint=False):
        super().__init__()
        from monai.networks.nets import SwinUNETR
        self.net = SwinUNETR(in_channels=in_ch, out_channels=out_ch, feature_size=feature_size,
                             depths=tuple(depths), num_heads=tuple(num_heads),
                             window_size=window_size, norm_name="instance",
                             dropout_path_rate=drop_path_rate, use_checkpoint=use_checkpoint,
                             spatial_dims=3)
        self.divisor = 2 ** 5  # patch_size 2 × four patch-merging stages

    def forward(self, x):
        xp, pads = pad_to_multiple(x, self.divisor, value=0.0)
        return crop_pads(self.net(xp), pads)

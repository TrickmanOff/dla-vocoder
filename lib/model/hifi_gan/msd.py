from typing import Any, Callable, Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MSDSub(nn.Module):
    def __init__(self, wave_scale: int = 1,
                 kernel_sizes: Sequence[int] = (15, 41, 41, 41, 41, 5),
                 strides: Sequence[int] = (1, 4, 4, 4, 4, 1),
                 hidden_channels: Sequence[int] = (16, 64, 256, 1024, 1024, 1024),
                 num_groups: Sequence[int] = (1, 4, 16, 64, 256, 1),
                 norm_fn: Callable[[nn.Module], nn.Module] = nn.utils.weight_norm,
                 act_cls: nn.Module = nn.LeakyReLU):
        # subdiscriminator architecture as in the MelGAN, in the HiFiGAN slightly different parameters are used
        super().__init__()
        assert len(kernel_sizes) == len(strides) == len(hidden_channels) == len(num_groups)

        self.wave_scale = wave_scale
        convs = nn.ModuleList()
        in_channels = 1
        for kernel_size, stride, out_channels, groups in \
                zip(kernel_sizes, strides, hidden_channels, num_groups):
            convs.append(nn.Sequential(
                norm_fn(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                  groups=groups)),
                act_cls(),
            ))
            in_channels = out_channels

        convs.append(
            norm_fn(nn.Conv1d(in_channels, 1, kernel_size=3))
        )
        self.convs = convs

    def forward(self, input: Tensor, **batch) -> Dict[str, Any]:
        """
        :param input: waveform of shape (B, 1, T)
        :return: {
            'output': of shape (B, num_features),
            'feature_maps': list of tensors
        }
        """
        if self.wave_scale != 1:
            input = F.avg_pool1d(input, kernel_size=self.wave_scale, stride=self.wave_scale)

        feature_maps = []
        for conv in self.convs:
            output = conv(input)
            feature_maps.append(output)
            input = output

        return {
            'output': feature_maps[-1].reshape(input.shape[0], -1),
            'feature_maps': feature_maps,
        }


class MSD(nn.Module):
    def __init__(self, wave_scales: Sequence[int] = (1, 2, 4),
                 **msd_sub_kwargs):
        super().__init__()
        self.subdiscriminators = nn.ModuleList(
            MSDSub(wave_scale=wave_scale,
                   norm_fn=nn.utils.parametrizations.spectral_norm if wave_scale == 1 else nn.utils.weight_norm,
                   **msd_sub_kwargs)
            for wave_scale in wave_scales
        )

    def forward(self, wave: Tensor, **batch) -> Dict[str, Any]:
        """
        :param wave: waveform of shape (B, 1, T)
        :return: {
            'outputs': list of tensors,
            'feature_maps': list of tensors
        }
        """
        feature_maps: List[Tensor] = []
        outputs: List[Tensor] = []
        for subdisc in self.subdiscriminators:
            result = subdisc(wave)
            outputs.append(result['output'])
            feature_maps += result['feature_maps']
        return {
            'outputs': outputs,
            'feature_maps': feature_maps,
        }

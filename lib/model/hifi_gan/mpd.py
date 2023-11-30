from typing import Any, Callable, Dict, List, Sequence

import torch
from torch import Tensor, nn


class MPDSub(nn.Module):
    def __init__(self, period: int,
                 padding_value: float = 0.,
                 norm_fn: Callable[[nn.Module], nn.Module] = nn.utils.weight_norm):
        super().__init__()

        convs = nn.ModuleList()
        in_channels = 1
        for l in range(1, 4+1):
            out_channels = 2**(5+l)
            convs.append(nn.Sequential(
                norm_fn(nn.Conv2d(in_channels, out_channels,
                                  kernel_size=(5, 1), stride=(3, 1))),
                nn.LeakyReLU(),
            ))
            in_channels = out_channels

        out_channels = 1024
        convs.append(nn.Sequential(
            norm_fn(nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=(5, 1))),
            nn.LeakyReLU(),
        ))
        in_channels = out_channels

        convs.append(
            norm_fn(nn.Conv2d(in_channels, out_channels=1, kernel_size=(3, 1)))
        )
        self.convs = convs

        self.period = period
        self.padding_value = padding_value

    def forward(self, input: Tensor, **batch) -> Dict[str, Any]:
        """
        :param input: waveform of shape (B, 1, T)
        :return:
        {
          'feature_maps': list of feature maps of subsequent convolutional layers
        so the output is the last value in the list
          'output': output of shape (B, features_dim)
        }
        """
        T = input.shape[-1]
        new_T = self.period * ((T + self.period - 1) // self.period)
        input = nn.functional.pad(input, (0, new_T - T), value=self.padding_value)
        B = input.shape[0]
        input = input.reshape(B, 1, -1, self.period)
        feature_maps = []
        for conv in self.convs:
            output = conv(input)
            feature_maps.append(output)
            # print(input.shape, '->', output.shape)
            input = output

        output = feature_maps[-1].reshape(B, -1)
        return {
            'output': output,
            'feature_maps': feature_maps,
        }


class MPD(nn.Module):
    def __init__(self, periods: Sequence[int] = (2, 3, 5, 7, 11), padding_value: float = 0.):
        super().__init__()
        self.subdiscriminators = nn.ModuleList(
            MPDSub(period=period, padding_value=padding_value)
            for period in periods
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

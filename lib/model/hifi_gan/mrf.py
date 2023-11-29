from typing import Any, Dict, Sequence

from torch import Tensor, nn


class ResSubBlock(nn.Module):
    def __init__(self, num_channels: int, kernel_size: int, dilations: Sequence[int]):
        super().__init__()
        seq = []
        for dilation in dilations:
            seq.append(nn.LeakyReLU())
            seq.append(nn.Conv1d(num_channels, num_channels,
                                 kernel_size=kernel_size, dilation=dilation,
                                 padding='same'))
        self.conv_seq = nn.Sequential(*seq)

    def forward(self, input: Tensor, **batch) -> Tensor:
        """
        :param input: of shape (B, num_channels, time)
        :return: of shape (B, num_channels, time)
        """
        return self.conv_seq(input) + input


class ResBlock(nn.Module):
    def __init__(self, num_channels: int, kernel_size: int, dilations: Sequence[Sequence[int]]):
        super().__init__()
        self.seq = nn.Sequential(*[
            ResSubBlock(num_channels=num_channels, kernel_size=kernel_size, dilations=dilation)
            for dilation in dilations
        ])

    def forward(self, input: Tensor, **batch) -> Tensor:
        """
        :param input: of shape (B, num_channels, time)
        :return: of shape (B, num_channels, time)
        """
        return self.seq(input)


class MRF(nn.Module):
    def __init__(self, num_channels: int, kernel_sizes: Sequence[int],
                 dilations: Sequence[Sequence[Sequence[int]]]):
        """
        :param kernel_sizes: k_r in the article
        :param dilations: D_r in the article
        """
        super().__init__()
        assert len(kernel_sizes) == len(dilations)
        self.blocks = nn.ModuleList(
            ResBlock(num_channels=num_channels, kernel_size=kernel_size, dilations=dilation)
            for kernel_size, dilation in zip(kernel_sizes, dilations)
        )

    def forward(self, input: Tensor, **batch) -> Tensor:
        """
        :param input: of shape (B, num_channels, time)
        :return: of shape (B, num_channels, time)
        """
        return sum(block(input) for block in self.blocks)


class MRFStack(nn.Module):
    def __init__(self, num_channels: int, transpose_kernel_sizes: Sequence[int], mrf_config: Dict[str, Any]):
        """
        :param num_channels: h_u
        :param transpose_kernel_sizes: k_u
        :param mrf_config:
        """
        super().__init__()

        out_channels = num_channels

        seq = []

        for transpose_kernel_size in transpose_kernel_sizes:
            assert out_channels % 2 == 0
            in_channels = out_channels
            out_channels //= 2
            assert transpose_kernel_size % 2 == 0

            seq += [
                nn.LeakyReLU(),
                nn.ConvTranspose1d(in_channels, out_channels,
                                   kernel_size=transpose_kernel_size,
                                   stride=transpose_kernel_size // 2,
                                   padding=transpose_kernel_size // 4),  # t_out = stride * t_in
                MRF(num_channels=out_channels, **mrf_config),
            ]

        self.seq = nn.Sequential(*seq)

    def forward(self, input: Tensor, **batch) -> Tensor:
        """
        :param input: of shape (B, C=num_channels, T)
        :return: (B, C_2, T)
        """
        return self.seq(input)

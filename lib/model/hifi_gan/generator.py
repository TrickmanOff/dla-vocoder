from typing import Any, Dict, Sequence

from torch import Tensor, nn

from .mrf import MRFStack
from lib.model.base_model import BaseModel


class HiFiGenerator(BaseModel):
    def __init__(self, mel_freqs_cnt: int, hidden_channels: int, transpose_kernel_sizes: Sequence[int],
                 mrf_config: Dict[str, Any]):
        """
        :param hidden_channels: h_u
        :param transpose_kernel_sizes: k_u
        :param mrf_config: {kernel_sizes: k_r, dilations: D_r}
        """
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(in_channels=mel_freqs_cnt, out_channels=hidden_channels,
                      kernel_size=7),
            MRFStack(num_channels=hidden_channels, transpose_kernel_sizes=transpose_kernel_sizes,
                     mrf_config=mrf_config),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=hidden_channels // (2**len(transpose_kernel_sizes)),
                      out_channels=1, kernel_size=7),
            nn.Tanh(),
        )

    def forward(self, mel_spec: Tensor, **batch) -> Tensor:
        """
        :param mel_spec of shape (B, freqs, T)
        :return: waveform of shape (B, 1, T' > T)
        """
        return self.seq(mel_spec)

from typing import Any, Dict, List, Optional

from .mpd import MPD
from .msd import MSD
from lib.model.base_model import BaseModel

import torch
from torch import Tensor


class HiFiDiscriminator(BaseModel):
    def __init__(self,
                 msd_config: Optional[Dict[str, Any]] = None,
                 mpd_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        msd_config = {} if msd_config is None else None
        mpd_config = {} if mpd_config is None else None

        self.msd = MSD(**msd_config)
        self.mpd = MPD(**mpd_config)

    def forward(self, wave: Tensor, **batch) -> Dict[str, Any]:
        """
        :param wave: waveform of shape (B, 1, T)
        :return: {
            'output': of shape (B, 1, num_features),
            'feature_maps': list of tensors
        }
        """
        feature_maps: List[Tensor] = []
        outputs: List[Tensor] = []
        for subdiscriminator in [self.msd, self.mpd]:
            result = subdiscriminator(wave)
            feature_maps += result['feature_maps']
            outputs.append(result['output'])
        return {
            'output': torch.concatenate(outputs, dim=-1),
            'feature_maps': feature_maps,
        }

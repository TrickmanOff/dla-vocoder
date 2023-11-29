from typing import List

import torch.nn.functional as F
from torch import Tensor

from lib.loss.base_loss import BaseLoss


class GeneratorFMLoss(BaseLoss):
    def forward(self, gen_disc_fmaps: List[Tensor], true_disc_fmaps: List[Tensor], **batch) -> Tensor:
        loss = 0.
        for gen_disc_fmap, true_disc_fmap in zip(gen_disc_fmaps, true_disc_fmaps):
            loss += F.l1_loss(gen_disc_fmap, true_disc_fmap, reduction='mean')
        return loss

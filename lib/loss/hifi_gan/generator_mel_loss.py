import torch.nn.functional as F
from torch import Tensor

from lib.loss.base_loss import BaseLoss


class GeneratorMelLoss(BaseLoss):
    def forward(self, gen_mel_spec: Tensor, mel_spec: Tensor, **batch) -> Tensor:
        return F.l1_loss(gen_mel_spec, mel_spec, reduction='mean')

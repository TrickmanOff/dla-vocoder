from typing import List

from torch import Tensor

from lib.loss.base_loss import BaseLoss


class DiscriminatorAdversarialLoss(BaseLoss):
    def forward(self, gen_disc_outputs: List[Tensor], true_disc_outputs: List[Tensor], **batch) -> Tensor:
        loss = 0.
        for gen_disc_output, true_disc_output in zip(gen_disc_outputs, true_disc_outputs):
            loss += ((true_disc_output - 1)**2).mean()
            loss += (gen_disc_output**2).mean()
        return loss

from torch import Tensor

from lib.loss.base_loss import BaseLoss


class GeneratorAdversarialLoss(BaseLoss):
    def forward(self, gen_disc_outputs: Tensor, **batch) -> Tensor:
        loss = 0.
        for gen_disc_output in gen_disc_outputs:
            loss += ((gen_disc_output - 1)**2).mean()
        return loss

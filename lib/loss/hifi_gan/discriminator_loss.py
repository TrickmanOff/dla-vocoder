from .discriminator_adv_loss import DiscriminatorAdversarialLoss
from lib.loss.combined_loss import CombinedLoss


class HiFiGANDiscriminatorLoss(CombinedLoss):
    def __init__(self, adv_loss_weight: float = 1.):
        losses = {
            'adv loss': DiscriminatorAdversarialLoss(),
        }
        weights = {
            'adv loss': adv_loss_weight,
        }
        super().__init__(losses, weights)

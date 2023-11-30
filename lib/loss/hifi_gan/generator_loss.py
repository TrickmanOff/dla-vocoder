from .generator_adv_loss import GeneratorAdversarialLoss
from .generator_mel_loss import GeneratorMelLoss
from .generator_fm_loss import GeneratorFMLoss
from lib.loss.combined_loss import CombinedLoss


class HiFiGANGeneratorLoss(CombinedLoss):
    def __init__(self, adv_loss_weight: float = 1.,
                 mel_loss_weight: float = 1.,
                 fm_loss_weight: float = 1.):
        losses = {
            'adv loss': GeneratorAdversarialLoss(),
            'mel loss': GeneratorMelLoss(),
            'fm loss': GeneratorFMLoss(),
        }
        weights = dict(
            zip(losses.keys(), (adv_loss_weight, mel_loss_weight, fm_loss_weight))
        )
        super().__init__(losses, weights)

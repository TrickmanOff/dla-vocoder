from .base_metric import BaseMetric
from lib.loss.hifi_gan.generator_mel_loss import GeneratorMelLoss


class MelLossMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_module = GeneratorMelLoss()

    def __call__(self, **batch):
        return self._loss_module(**batch)

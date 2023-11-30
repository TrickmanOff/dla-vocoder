import random
from dataclasses import dataclass
from random import shuffle
from typing import Dict, Sequence, Optional

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from lib.postprocessing.base_postprocessor import BasePostprocessor
from lib.trainer.base_trainer import BaseTrainer
from lib.trainer.optimization_stepper import OptimizationStepper
from lib.mel import MelSpectrogram
from lib.metric.base_metric import BaseMetric
from lib.model.base_model import BaseModel
from lib.logger.utils import plot_spectrogram_to_buf
from lib.loss.base_loss import BaseLoss
from lib.utils import inf_loop, MetricTracker, get_lr, align_last_dim


@dataclass
class TrainedNeuralNetwork:
    model: BaseModel
    optimization_stepper: OptimizationStepper
    loss_module: BaseLoss


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(
            self,
            generator: TrainedNeuralNetwork,
            discriminator: TrainedNeuralNetwork,
            metrics: Sequence[BaseMetric],
            config,
            device,
            dataloaders,
            mel_spec_gen: MelSpectrogram,
            postprocessor: Optional[BasePostprocessor] = None,
            len_epoch=None,
            skip_oom=True,
            n_critic: int = 1
    ):
        super().__init__(
            models={'generator': generator.model, 'discriminator': discriminator.model},
            metrics=metrics,
            optimizers={'generator': generator.optimization_stepper, 'discriminator': discriminator.optimization_stepper},
            config=config,
            device=device,
        )
        self.losses = {
            'generator': generator.loss_module,
            'discriminator': discriminator.loss_module,
        }
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.postprocessor = postprocessor
        self.mel_spec_gen = mel_spec_gen
        self.log_step = config["trainer"].get("log_step", 50)

        self.train_metrics = {
            "discriminator": MetricTracker(
                "loss", "grad norm",
                *discriminator.loss_module.get_loss_parts_names(),
                *[m.name for m in self.metrics if m.calc_on_train],
                writer=self.writer
            ),
            "generator": MetricTracker(
                "loss", "grad norm",
                *generator.loss_module.get_loss_parts_names(),
                *[m.name for m in self.metrics if m.calc_on_train],
                writer=self.writer
            )
        }
        self.evaluation_metrics = MetricTracker(
            *[m.name for m in self.metrics if m.calc_on_non_train],
            writer=self.writer
        )
        # self.accumulated_grad_steps = 0
        # self.accumulate_grad_steps = config["trainer"].get("accumulate_grad_steps", 1)
        self.n_critic = n_critic

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["mel_spec", "target_wave"]:
            if tensor_for_gpu in batch:
                batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            for model in self.models.values():
                clip_grad_norm_(
                    model.parameters(), self.config["trainer"]["grad_norm_clip"]
                )

    def safe_process_batch(self, *args, **kwargs) -> Optional[Dict]:
        try:
            batch = self.process_batch(
                *args, **kwargs
            )
            return batch
        except RuntimeError as e:
            if "out of memory" in str(e) and self.skip_oom:
                self.logger.warning("OOM on batch. Skipping batch.")
                for network in self.models.values():
                    for p in network.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                torch.cuda.empty_cache()
                return None
            else:
                raise e

    def _train_epoch(self, epoch) -> Dict:
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        for name in self.models:
            self.models[name].train()
            self.losses[name].train()
            self.train_metrics[name].reset()
        self.writer.add_scalar("epoch", epoch)
        iter_dataloader = iter(self.train_dataloader)
        for batch_idx in tqdm(range(self.len_epoch), desc="train"):
            discriminator_loss = None
            for _ in range(self.n_critic):
                batch = next(iter_dataloader)
                train_metrics = self.train_metrics["discriminator"]
                batch = self.safe_process_batch(batch=batch, is_train=True, metrics=train_metrics,
                                                is_discriminator=True)
                if batch is None:
                    continue
                discriminator_loss = batch["loss"].item()
                train_metrics.update("grad norm", self.get_grad_norm(self.models["discriminator"]))

            batch = next(iter_dataloader)
            train_metrics = self.train_metrics["generator"]
            batch = self.safe_process_batch(batch=batch, is_train=True, metrics=train_metrics,
                                            is_discriminator=False)
            if batch is None:
                continue
            train_metrics.update("grad norm", self.get_grad_norm(self.models["generator"]))

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Generator loss: {:.6f} Discriminator loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item(), discriminator_loss,
                    )
                )

                self._log_predictions(**batch)

                for name in self.models:
                    self.writer.add_scalar(
                        f"{name} learning rate",
                        self.optimizers[name].get_lr()
                    )

                    self._log_scalars(self.train_metrics[name], prefix=f'{name} ')
                    # we don't want to reset train metrics at the start of every epoch
                    # because we are interested in recent train metrics
                    last_train_metrics = self.train_metrics[name].result()
                    self.train_metrics[name].reset()

            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        for optimizer in self.optimizers.values():
            optimizer.epoch_finished()

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker,
                      is_discriminator: bool = True):
        if not is_train:
            assert not is_discriminator, 'Only generator is used for validation'
        batch = self.move_batch_to_device(batch, self.device)
        trained_model_name = 'discriminator' if is_discriminator else 'generator'
        self.models[trained_model_name].requires_grad_(True)
        self.models['generator' if is_discriminator else 'discriminator'].requires_grad_(False)
        self.optimizers[trained_model_name].optimizer.zero_grad()

        gen_wave = self.models['generator'](**batch)
        batch['gen_wave'] = gen_wave
        batch['target_wave'] = align_last_dim(batch['target_wave'], gen_wave)

        if is_train:
            gen_disc_result = self.models['discriminator'](wave=batch['gen_wave'])
            true_disc_result = self.models['discriminator'](wave=batch['target_wave'])

            batch['gen_disc_outputs'] = gen_disc_result['outputs']
            batch['gen_disc_fmaps'] = gen_disc_result['feature_maps']
            batch['true_disc_outputs'] = true_disc_result['outputs']
            batch['true_disc_fmaps'] = true_disc_result['feature_maps']

        if not is_discriminator:
            batch['gen_mel_spec'] = self.mel_spec_gen(batch['gen_wave']).squeeze(1)
            batch['mel_spec'] = align_last_dim(batch['mel_spec'], batch['gen_mel_spec'],
                                               padding_value=self.mel_spec_gen.config.pad_value)

        loss_module = self.losses[trained_model_name]
        # criterion returns a dict, in which the final loss has a key 'loss'

        if is_train:  # in the current solution I don't need the all values of loss on validation data
            losses = loss_module(**batch)
            batch.update(losses)
            losses['loss'].backward()
            for loss_part in losses:
                metrics.update(loss_part, batch[loss_part].item())

        if self.postprocessor is not None:
            batch = self.postprocessor(**batch)
        if is_train:
            self._clip_grad_norm()
            self.optimizers[trained_model_name].step()

        for met in self.metrics:
            if (is_train and met.calc_on_train) or (not is_train and met.calc_on_non_train):
                metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """=
        Validate after training an epoch
        Use only generator here!

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        for name in self.models:
            self.models[name].eval()
            self.losses[name].eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                    is_discriminator=False,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(**batch)
            # self._log_spectrogram(batch["spectrogram"])

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            mel_spec,
            gen_mel_spec,
            mel_length,
            gen_wave,
            target_wave,
            target_wave_length,
            id=None,
            examples_to_log=3,
            *args,
            **kwargs,
    ):
        if self.writer is None:
            return

        batch_size = len(mel_spec)
        # pred_waves = self.spectrogram_decoder.decode_as_wave(pred_mel_spec)  # (B, T)
        # true_waves = self.spectrogram_decoder.decode_as_wave(true_mel_spec) if true_mel_spec is not None else [None] * batch_size

        if id is None:
            id = range(batch_size)

        tuples = list(zip(id, mel_spec, gen_mel_spec, mel_length, gen_wave, target_wave, target_wave_length))
        shuffle(tuples)
        rows = {}

        for id, mel_spec, gen_mel_spec, mel_len, gen_wave, target_wave, target_wave_len \
                in tuples[:examples_to_log]:
            rows[id] = {
                'true_mel_spec': self._create_image_for_writer(mel_spec[:, :mel_len]),
                'gen_mel_spec': self._create_image_for_writer(gen_mel_spec[:, :mel_len]),
                'true_wave': self._create_audio_for_writer(target_wave, target_wave_len),
                'gen_wave': self._create_audio_for_writer(gen_wave, target_wave_len),
            }

        table = pd.DataFrame.from_dict(rows, orient="index")\
                            .reset_index().rename(columns={'index': 'id'})
        self.writer.add_table("predictions", table)

    def _create_audio_for_writer(self, audio: torch.Tensor, length=None):
        audio = audio.detach().cpu().squeeze()
        if length is not None:
            audio = audio[:length]
        return self.writer.create_audio(audio, sample_rate=self.mel_spec_gen.config.sr)

    def _create_image_for_writer(self, image: torch.Tensor):
        image = image.detach().cpu()
        image = PIL.Image.open(plot_spectrogram_to_buf(image))
        return self.writer.create_image(image)

    def _log_stats(self, **batch):
        idx = random.randint(0, batch['true_pitch'].shape[0] - 1)
        self.writer.add_multiline_plot('pitch', {'true pitch': batch['true_pitch'][idx],
                                                 'pred pitch': batch['pred_pitch'][idx]})
        self.writer.add_multiline_plot('energy', {'true energy': batch['true_energy'][idx],
                                                  'pred energy': batch['pred_energy'][idx]})
        pred_mel_spec_img = PIL.Image.open(plot_spectrogram_to_buf(batch['pred_mel_spec'][idx]))
        true_mel_spec_img = PIL.Image.open(plot_spectrogram_to_buf(batch['true_mel_spec'][idx]))
        self.writer.add_image("pred spectrogram", ToTensor()(pred_mel_spec_img))
        self.writer.add_image("true spectrogram", ToTensor()(true_mel_spec_img))

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, model: BaseModel, norm_type=2):
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker, prefix: str = ''):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{prefix}{metric_name}", metric_tracker.avg(metric_name))

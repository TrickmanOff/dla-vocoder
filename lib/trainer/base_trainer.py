from abc import abstractmethod
from typing import Dict, Iterable

import torch
from numpy import inf

from lib.metric.base_metric import BaseMetric
from lib.model.base_model import BaseModel
from lib.logger import get_visualizer
from lib.storage.experiments_storage import RunStorage
from lib.storage.external_storage import ExternalStorage
from lib.trainer.optimization_stepper import OptimizationStepper


class BaseTrainer:
    """
    Base class for all trainers
    Responsible for saving and loading checkpoints
    """
    def __init__(self, models: Dict[str, BaseModel], metrics: Iterable[BaseMetric], optimizers: Dict[str, OptimizationStepper], config, device):
        self.device = device
        self.config = config
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])

        self.models = models
        self.metrics = metrics
        self.optimizers = optimizers

        # for interrupt saving
        self._last_epoch = 0

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.monitor = cfg_trainer.get("monitor", "off")

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.run_storage: RunStorage = config.run_storage
        self.external_storage: ExternalStorage = config.external_storage

        # setup visualization writer instance
        self.writer = get_visualizer(
            config, self.logger, cfg_trainer["visualize"]
        )

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch) -> dict:
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError()

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not,
                    # according to specified metric(mnt_metric)
                    if self.mnt_mode == "min":
                        improved = log[self.mnt_metric] <= self.mnt_best
                    elif self.mnt_mode == "max":
                        improved = log[self.mnt_metric] >= self.mnt_best
                    else:
                        improved = False
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        models = {}
        for model_name, model in self.models.items():
            models[model_name] = {
                "arch": type(model).__name__,
                "state_dict": model.state_dict(),
            }

        optimizers = {}
        for model_name, optim_stepper in self.optimizers.items():
            optimizers[model_name] = optim_stepper.state_dict()

        state = {
            "epoch": epoch,
            "monitor_best": self.mnt_best,
            "config": self.config,
            "models": models,
            "optimizers": optimizers,
        }

        if not (only_best and save_best):
            checkpoint_name = "checkpoint-epoch{}".format(epoch)
            self.run_storage.save_checkpoint(checkpoint_name, state)
            # if self.external_storage is not None:
            #     self.external_storage.export_checkpoint(self.run_storage, checkpoint_name)
        if save_best:
            checkpoint_name = "model_best"
            self.run_storage.save_checkpoint(checkpoint_name, state)
            if self.external_storage is not None:
                self.external_storage.export_checkpoint(self.run_storage, checkpoint_name)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        for model_name, model_data in checkpoint["models"].items():
            if checkpoint["config"][model_name]["arch"] != self.config[model_name]["arch"]:
                self.logger.warning(
                    "Warning: Architecture configuration given in config file is different from that "
                    "of checkpoint. This may yield an exception while state_dict is being loaded."
                )
            self.models[model_name].load_state_dict(model_data["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        for model_name, state_dict in checkpoint["optimizers"].items():
            if (
                    checkpoint["config"][model_name]["optimizer"] != self.config[model_name]["optimizer"] or
                    checkpoint["config"][model_name].get("lr_scheduler", None) != self.config[model_name].get("lr_scheduler", None)
            ):
                self.logger.warning(
                    "Warning: Optimizer or lr_scheduler given in config file is different "
                    "from that of checkpoint. Optimizer parameters not being resumed."
                )
            else:
                self.optimizers[model_name].load_state_dict(state_dict)

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )

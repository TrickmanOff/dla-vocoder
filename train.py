import argparse
import collections
import warnings
from copy import deepcopy

import numpy as np
import torch

import lib.loss as module_loss
import lib.metric as module_metric
import lib.model as module_arch
import lib.postprocessing as module_postprocessing
from lib.config_processing.parse_config import ConfigParser
from lib.trainer import Trainer
from lib.mel import MelSpectrogram
from lib.utils import prepare_device
from lib.utils.object_loading import get_dataloaders, get_trained_network


warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)


def main(config: ConfigParser):
    logger = config.get_logger("train")

    # sampling_rate = config["common"]["sr"]

    mel_spec_gen = MelSpectrogram(**config.config.get("mel_spectrogram_generator", {}))

    # setup data_loader instances
    dataloaders = get_dataloaders(config, mel_spec_gen=mel_spec_gen)

    # build model architecture, then print to console
    generator = get_trained_network(config, "generator",
                                    model_kwargs={"mel_freqs_cnt": mel_spec_gen.config.n_mels})
    discriminator = get_trained_network(config, "discriminator")

    device, device_ids = prepare_device(config["n_gpu"])
    for trained_net in [generator, discriminator]:
        logger.info(trained_net.model)

        # prepare for (multi-device) GPU training
        trained_net.model = trained_net.model.to(device)
        if len(device_ids) > 1:
            trained_net.model = torch.nn.DataParallel(trained_net.model, device_ids=device_ids)

    metrics = [
        config.init_obj(metric_dict, module_metric)
        for metric_dict in config.config.get("metrics", [])
    ]

    if "postprocessor" in config.config:
        postprocessor = config.init_obj(config["postprocessor"], module_postprocessing)
    else:
        postprocessor = None

    trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        metrics=metrics,
        config=config,
        device=device,
        dataloaders=dataloaders,
        mel_spec_gen=deepcopy(mel_spec_gen).to(device),
        postprocessor=postprocessor,
        len_epoch=config["trainer"].get("len_epoch", None),
        n_critic=config["trainer"].get("n_critic", 1),
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="TSS trainer")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

from operator import xor
from typing import Optional

import torch
from torch.utils.data import ConcatDataset, DataLoader

import lib.datasets
# from lib import batch_sampler as batch_sampler_module
from lib.collate_fn.collate import Collator
from lib.config_processing.parse_config import ConfigParser
from lib.mel import MelSpectrogram
from lib.trainer.trainer import OptimizationStepper, TrainedNeuralNetwork
# from lib.text_encoder.base_encoder import BaseTextEncoder


def get_dataloaders(configs: ConfigParser, mel_spec_gen: Optional[MelSpectrogram] = None):
    dataloaders = {}
    for split, params in configs["data"].items():
        if not isinstance(params, dict):
            continue
        num_workers = params.get("num_workers", 1)

        if split == 'train':
            drop_last = True
        else:
            drop_last = False

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(configs.init_obj(ds, lib.datasets, config_parser=configs))
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert xor("batch_size" in params, "batch_sampler" in params), \
            "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
            batch_sampler = None
        # elif "batch_sampler" in params:
        #     batch_sampler = configs.init_obj(params["batch_sampler"], batch_sampler_module,
        #                                      data_source=dataset)
        #     bs, shuffle = 1, False
        else:
            raise Exception()

        # Fun fact. An hour of debugging was wasted to write this line
        assert bs <= len(dataset), \
            f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        collator = Collator(mel_spec_gen=mel_spec_gen)

        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=collator,
            shuffle=shuffle, num_workers=num_workers,
            batch_sampler=batch_sampler, drop_last=drop_last
        )
        dataloaders[split] = dataloader
    return dataloaders


def get_trained_network(config: ConfigParser, name,
                        model_kwargs=None,
                        optim_kwargs=None,
                        scheduler_kwargs=None,
                        loss_kwargs=None,
                        module_arch=lib.model,
                        module_optim=torch.optim,
                        module_scheduler=torch.optim.lr_scheduler,
                        module_loss=lib.loss) -> TrainedNeuralNetwork:
    network_config = config[name]
    model = config.init_obj(network_config["arch"], module_arch, **(model_kwargs or {}))
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj(network_config["optimizer"], module_optim, trainable_params, **(optim_kwargs or {}))
    if "lr_scheduler" in network_config:
        lr_scheduler = config.init_obj(network_config["lr_scheduler"]["scheduler"], module_scheduler, optimizer, **(scheduler_kwargs or {}))
        scheduler_mode = network_config["lr_scheduler"]["scheduler_mode"]
    else:
        lr_scheduler = None
        scheduler_mode = None

    optimization_stepper = OptimizationStepper(optimizer, lr_scheduler, scheduler_mode=scheduler_mode)

    loss_module = config.init_obj(config[name]["loss"], module_loss, **(loss_kwargs or {}))

    return TrainedNeuralNetwork(
        model=model,
        optimization_stepper=optimization_stepper,
        loss_module=loss_module,
    )

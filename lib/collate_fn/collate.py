import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, LongTensor

from lib.mel import MelSpectrogram


logger = logging.getLogger(__name__)


PADDING_VALUE = 0


def pad_last_d(input: List[Tensor], padding_value: float = PADDING_VALUE) -> Tuple[Tensor, LongTensor]:
    """
    each of B inputs is of shape (..., S_i)

    result:
        stack:  (B, ..., max_i S_i)
        length: (B,) - initial lengths of each sequence
    """
    length = LongTensor([x.shape[-1] for x in input])
    max_len = length.max()

    shape = [len(input)] + list(input[0].shape)
    shape[-1] = max_len

    stack = torch.full(shape, padding_value, dtype=input[0].dtype)

    for i, x in enumerate(input):  # (..., S_i)
        stack[i, ..., :x.shape[-1]] = x

    return stack, length


class Collator:
    def __init__(self, mel_spec_gen: Optional[MelSpectrogram] = None,
                 mel_silence_value: float = 0.):
        self.mel_spec_gen = mel_spec_gen
        if mel_spec_gen is not None:
            mel_silence_value = mel_spec_gen.config.pad_value
        self.mel_silence_value = mel_silence_value

    def __call__(self, dataset_items: List[dict]) -> Dict[str, Any]:
        """
            Collate and pad fields in dataset items
            """
        all_items = defaultdict(list)  # {str: [val1, val2, ...], ...}
        for items in dataset_items:
            for key, val in items.items():
                all_items[key].append(val)

        result_batch = {}

        result_batch['id'] = all_items['id']

        # wave
        result_batch['target_wave'], result_batch['target_wave_length'] = pad_last_d(
            all_items['wave'])

        # mel
        if 'mel' in all_items:
            result_batch['mel_spec'], result_batch['mel_length'] = pad_last_d(all_items['mel'],
                                                                              padding_value=self.mel_silence_value)
        else:
            result_batch['mel_spec'] = self.mel_spec_gen(result_batch['target_wave']).squeeze(1)
            result_batch['mel_length'] = self.mel_spec_gen.transform_length(result_batch['target_wave_length'])

        return result_batch

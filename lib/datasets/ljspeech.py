import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torchaudio
from speechbrain.utils.data_utils import download_file
from torch import Tensor

from .base_dataset import BaseDataset


class LJSpeechDataset(BaseDataset):
    URL_LINK = 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'

    SAMPLE_RATE = 22050
    VAL_NUM_SAMPLES = 100

    def __init__(self, data_dir: Union[str, Path],
                 mel_spec_fn: Callable[[Tensor], Tensor],
                 indices_dir: Optional[Union[str, Path]] = None,
                 max_wave_time_samples: Optional[int] = None,
                 train: bool = True):
        self._data_dir = Path(data_dir)
        indices_dir = Path(data_dir if indices_dir is None else indices_dir)
        self._indices_dir = indices_dir

        self._index = self._get_index(train)
        self._train = train
        self._max_wave_time_samples = max_wave_time_samples
        self._mel_spec_fn = mel_spec_fn

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """
        :return: {
            'id': str,
            'wave': (1, T),
            'mel': (1, freqs, T'),
        }
        """
        wav_filepath = self._index[item]
        wave, sr = torchaudio.load(wav_filepath)  # wave of shape (1, T)
        assert sr == self.sample_rate
        if self._train and self._max_wave_time_samples is not None and wave.shape[1] > self._max_wave_time_samples:
            st_time = random.randint(0, wave.shape[1] - self._max_wave_time_samples)
            wave = wave[:, st_time:st_time+self._max_wave_time_samples]
        mel_spec = self._mel_spec_fn(wave)  # (1, freqs, T)
        return {
            'id': Path(wav_filepath).stem,
            'wave': wave,
            'mel': mel_spec,
        }

    def _get_index(self, train: bool) -> List[str]:
        index_filepath = self._indices_dir / 'ljspeech_index.json'
        if not index_filepath.exists():
            self._build_index(index_filepath)
        index = json.load(open(index_filepath, 'r'))
        random.seed(42)
        random.shuffle(index)
        if train:
            return index[:-self.VAL_NUM_SAMPLES]
        else:
            return index[-self.VAL_NUM_SAMPLES:]

    def _build_index(self, index_filepath: Path):
        dataset_dirpath = self._data_dir / 'LJSpeech-1.1'
        if not dataset_dirpath.exists():
            self._download_dataset()
        wavs_dirpath = dataset_dirpath / 'wavs'

        print('Building audio index...')
        audio_pattern = re.compile(r'^LJ\d{3}-\d{4}\.wav$')
        audio_filenames = sorted(
            filename for filename in os.listdir(wavs_dirpath)
            if audio_pattern.match(filename))

        full_index = [str(wavs_dirpath / filename) for filename in audio_filenames]
        index_filepath.parent.mkdir(parents=True, exist_ok=True)
        json.dump(full_index, open(index_filepath, 'w'))

    def _download_audios(self):
        arch_filepath = self._data_dir / 'LJSpeech-1.1.tar.bz2'
        extracted_dataset_dirpath = self._data_dir / 'LJSpeech-1.1'
        self._download_archive(self.URL_LINK, arch_filepath, extracted_dataset_dirpath,
                               desc='audios')

    @staticmethod
    def _download_archive(link: str, arch_filepath: Path, extracted_dirpath: Path, desc: str):
        if not extracted_dirpath.exists():
            if not arch_filepath.exists():
                print(f'Downloading LJSpeech {desc}...')
                download_file(link, arch_filepath)
            print(f'Extracting LJSpeech {desc}...')
            shutil.unpack_archive(arch_filepath, extracted_dirpath.parent)
            # os.remove(arch_filepath)

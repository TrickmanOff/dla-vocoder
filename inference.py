import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
from speechbrain.utils.data_utils import download_file
from torch import nn
from tqdm import tqdm

import lib.model as module_model
from lib.config_processing import ConfigParser
from lib.mel import MelSpectrogram


URL_LINKS = {
    'gen_checkpoint': 'https://www.googleapis.com/drive/v3/files/1eXQNxni3gdvHJ9WoOvvSutyZ4l9sw5qa?alt=media&key=AIzaSyBAigZjTwo8uh77umBBmOytKc_qjpTfRjI',
}

CONFIG_FILEPATH = 'configs/kaggle/basic_orig_loss.json'
INFERENCE_DIRPATH = 'inference'


def main(config: ConfigParser, input_dirpath: Path, output_dirpath: Path,
         checkpoint_filepath: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # download model
    if not checkpoint_filepath.exists():
        print('Downloading model checkpoint...')
        download_file(URL_LINKS['gen_checkpoint'], checkpoint_filepath)

    # initialize Mel Spectrogram Generator
    mel_spec_gen = MelSpectrogram()

    # initialize model
    model: nn.Module = config.init_obj(config["generator"]["arch"], module_model,
                                       mel_freqs_cnt=mel_spec_gen.config.n_mels)
    state_dict = torch.load(checkpoint_filepath, map_location=device)["generator_state_dict"]
    model.load_state_dict(state_dict)
    print('Model weights loaded, ready to roll')
    model.eval()

    # process audios
    output_dirpath.mkdir(parents=True, exist_ok=True)
    for filename in tqdm(os.listdir(input_dirpath), desc='Processing audios'):
        # only wavs are supported
        if os.path.splitext(filename)[1] != '.wav':
            continue
        wav_filepath = input_dirpath / filename
        wave, sr = torchaudio.load(wav_filepath)  # (1, T)
        assert sr == mel_spec_gen.config.sr

        mel_spec = mel_spec_gen(wave)  # (1, n_mels, T')
        output = model(mel_spec)  # (1, 1, T'')
        output_wave = output[0]
        # assert wave.shape == output_wave.shape, f'{wave.shape} != {output_wave.shape}'

        output_filepath = output_dirpath / (os.path.splitext(filename)[0] + '_gen.wav')
        torchaudio.save(output_filepath, output_wave, sr)
    print('All audios have been processed')


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="TSS trainer")
    args.add_argument(
        "-i",
        "--input",
        default='demo_audios',
        type=str,
        help="a directory with audios",
    )
    args.add_argument(
        "-o",
        "--output",
        default=os.path.join(INFERENCE_DIRPATH, 'result'),
        type=str,
        help="a directory with resulting audios",
    )
    args.add_argument(
        "-c",
        "--checkpoint",
        default=os.path.join(INFERENCE_DIRPATH, 'checkpoint.pth'),
        type=str,
        help="model checkpoint filepath",
    )
    args = args.parse_args()

    config_dict = json.load(open(CONFIG_FILEPATH, 'r'))
    if "external_storage" in config_dict["trainer"]:
        config_dict["trainer"].pop("external_storage")
    config = ConfigParser(config_dict)
    main(config, Path(args.input), Path(args.output), Path(args.checkpoint))

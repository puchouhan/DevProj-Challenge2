import os
import random
import numpy as np
import zipfile
import requests
import io
from functools import partial
from sklearn.model_selection import train_test_split

import torch
from torch.utils import data
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import config
from dataset import transforms


def download_extract_zip(url, file_path):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise ConnectionError(f"Failed to download {url}. Error: {response.status_code}")

    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, 'wb') as fd:
        for chunk in response.iter_content(chunk_size=128 * 1024):
            fd.write(chunk)

    with zipfile.ZipFile(file_path, 'r') as z:
        z.extractall(dir_path)


class ESC50(data.Dataset):

    def __init__(self, root, test_folds=frozenset((1,)), subset="train", global_mean_std=(0.0, 0.0), download=False):
        self.cache = {}
        audio = 'ESC-50-master/audio'
        root = os.path.normpath(root)
        audio = os.path.join(root, audio)
        if subset in {"train", "test", "val"}:
            self.subset = subset
        else:
            raise ValueError
        # path = path.split(os.sep)
        if not os.path.exists(audio) and download:
            os.makedirs(root, exist_ok=True)
            file_name = 'master.zip'
            file_path = os.path.join(root, file_name)
            url = f'https://github.com/karoldvl/ESC-50/archive/{file_name}'
            download_extract_zip(url, file_path)

        self.root = audio
        # getting name of all files inside the all the train_folds
        temp = sorted(os.listdir(self.root))
        folds = {int(v.split('-')[0]) for v in temp}
        self.test_folds = set(test_folds)
        self.train_folds = folds - test_folds
        train_files = [f for f in temp if int(f.split('-')[0]) in self.train_folds]
        test_files = [f for f in temp if int(f.split('-')[0]) in test_folds]
        # sanity check
        assert set(temp) == (set(train_files) | set(test_files))
        if subset == "test":
            self.file_names = test_files
        else:
            if config.val_size:
                train_files, val_files = train_test_split(train_files, test_size=config.val_size, random_state=0)
            if subset == "train":
                self.file_names = train_files
            else:
                self.file_names = val_files
        # the number of samples in the wave (=length) required for spectrogram
        out_len = int(((config.sr * 5) // config.hop_length) * config.hop_length)
        train = self.subset == "train"
        if train:
            self.wave_transforms = transforms.Compose(
                torch.Tensor,
                transforms.RandomNoise(min_noise=0.001, max_noise=0.005),  # Nur im Training
                transforms.RandomScale(max_scale=1.15),  # Nur im Training
                transforms.RandomPadding(out_len=out_len, train=True),
                transforms.RandomCrop(out_len=out_len, train=True)
            )
            self.spec_transforms = transforms.Compose(
                torch.Tensor,
                partial(torch.unsqueeze, dim=0),
                transforms.FrequencyMask(max_width=12, numbers=2),  # Nur im Training
                transforms.TimeMask(max_width=15, numbers=2)  # Nur im Training
            )
        else:  # Für Validierung/Test
            self.wave_transforms = transforms.Compose(
                torch.Tensor,
                # Kein RandomNoise, kein RandomScale
                transforms.RandomPadding(out_len=out_len, train=False),  # Deterministisches Padding
                transforms.RandomCrop(out_len=out_len, train=False)  # Deterministischer Crop
            )
            self.spec_transforms = transforms.Compose(
                torch.Tensor,
                partial(torch.unsqueeze, dim=0)
                # Kein FrequencyMask, kein TimeMask
            )
        self.global_mean = global_mean_std[0]
        self.global_std = global_mean_std[1]
        self.n_mfcc = config.n_mfcc if hasattr(config, "n_mfcc") else None

        # Mel-Spektrogramm-Transformation
        self.melspec_transform = T.MelSpectrogram(
            sample_rate=config.sr,
            n_fft=1024,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            power=2.0,
        )

        # MFCC-Transformation
        if self.n_mfcc:
            self.mfcc_transform = T.MFCC(
                sample_rate=config.sr,
                n_mfcc=self.n_mfcc,
                melkwargs={
                    'n_fft': 1024,
                    'hop_length': config.hop_length,
                    'n_mels': config.n_mels,
                }
            )

        # AmplitudeToDB-Transformation
        self.amplitude_to_db = T.AmplitudeToDB()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        if index in self.cache:
            wave_copy, class_id = self.cache[index]
        else:
            path = os.path.join(self.root, file_name)

            # torchaudio zum Laden verwenden
            waveform, sample_rate = torchaudio.load(path)

            # Resampling falls nötig
            if sample_rate != config.sr:
                wave = F.resample(waveform, sample_rate, config.sr)
            else:
                wave = waveform

            # Umwandlung in Mono falls Stereo
            if wave.shape[0] > 1:  # Stereo zu Mono
                wave = torch.mean(wave, dim=0, keepdim=True)

            # identifying the label of the sample from its name
            temp = file_name.split('.')[0]
            class_id = int(temp.split('-')[-1])

            # normalizing waves to [-1, 1]
            if torch.abs(wave).max() > 1.0:
                wave = (wave - wave.min()) / (wave.max() - wave.min()) * 2.0 - 1.0
            wave = wave * 32768.0

            # Remove silent sections
            indices = torch.nonzero(wave)
            if indices.numel() > 0:
                start = indices[0]
                end = indices[-1]
                wave = wave[:, start:end + 1]

            # Für die Konsistenz mit dem Rest des Codes
            wave_copy = wave.clone()
            self.cache[index] = (wave_copy, class_id)

        # Anwendung der Wellenform-Transformationen
        wave_copy = self.wave_transforms(wave_copy)

        # Achse für librosa-Kompatibilität entfernen (da wir nun mit Tensoren arbeiten)
        if wave_copy.dim() > 1:
            wave_copy = wave_copy.squeeze(0)

        # Feature-Extraktion mit torchaudio
        if self.n_mfcc:
            # MFCC mit torchaudio berechnen
            feat = self.mfcc_transform(wave_copy)
            # torchaudio gibt MFCC im Format [1, n_mfcc, n_frames] zurück,
            # wir entfernen die erste Dimension für die Kompatibilität
            feat = feat.squeeze(0)
        else:
            # Mel-Spektrogramm mit torchaudio berechnen
            s = self.melspec_transform(wave_copy)
            # Umwandlung in dB-Skala
            log_s = self.amplitude_to_db(s)
            # Entfernen der ersten Dimension für die Kompatibilität
            log_s = log_s.squeeze(0)

            # Masking für Spektrogramme anwenden
            feat = self.spec_transforms(log_s)

        # Normalisierung
        if self.global_mean:
            feat = (feat - self.global_mean) / self.global_std

        return file_name, feat, class_id

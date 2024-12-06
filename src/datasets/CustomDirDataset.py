import json
import logging
import os
import shutil
from curses.ascii import isascii
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from src.utils.io_utils import ROOT_PATH


class CustomTextDirDataset:
    def __init__(self, path):
        self.path = ROOT_PATH / path

    def get_texts(self):
        texts = []
        for path in (self.path / "transcriptions").iterdir():
            assert path.suffix == ".txt"

            sample = {}
            with path.open() as file:
                sample["text"] = file.read()
            sample["filename"] = path.stem

            texts.append(sample)
        return texts


class CustomAudioDirDataset:
    def __init__(self, path):
        self.path = ROOT_PATH / path

    def get_audios(self):
        gt_audios = []
        for path in (self.path / "utterences").iterdir():
            assert path.suffix in [".mp3", ".wav", ".flac", ".m4a"]

            sample = {}
            target_sr = 22050
            audio, sr = torchaudio.load(path)
            audio = audio[0:1, :]
            if sr != target_sr:
                audio = torchaudio.functional.resample(audio, sr, target_sr)
            sample["gt_audio"] = audio
            sample["filename"] = path.stem

            gt_audios.append(sample)
        return gt_audios

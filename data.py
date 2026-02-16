import os
from pathlib import Path
import random
import sys

from datasets import load_dataset
import numpy as np
from pydub import AudioSegment
import torch
from torch.utils.data import IterableDataset, Dataset

from audio_preprocessor import *
from augment_data import *

if sys.platform == "win32":
    os.add_dll_directory(r"C:\Users\eacha\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1.1-full_build-shared\bin")

SAMPLE_FREQ = 16000
COEF_PER_FRAME = 12
FRAME_LENGTH = 400
FRAME_STEP = 300
FRAMES_PER_SEC = len(range(0, SAMPLE_FREQ - FRAME_LENGTH, FRAME_STEP)) # 40
INPUT_SIZE = COEF_PER_FRAME * FRAMES_PER_SEC

with open("token.txt") as f:
    token = f.read()

class CombinedDataset(IterableDataset):
    def __init__(self, split="train",
                 neg1="clean", neg2="dirty",
                 txt_path="train_pos_set.txt", txt_path_neg="train_neg_set.txt",
                 batches=2000, batch_size=64):
        super().__init__()
        self.neg_ds = PeoplesSpeech(neg1, split)
        self.neg_ds2 = PeoplesSpeech(neg2, split)
        self.neg_set3 = WavDataset(txt_path_neg)
        self.pos_set = WavDataset(txt_path)
        self.noise_set = NoiseSet()
        self.len = batches * batch_size

        self.neg_set = iter(self.neg_ds)
        self.neg_set2 = iter(self.neg_ds2)
    
    def __len__(self):
        return self.len
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        l = self.len // worker_info.num_workers
        for _ in range(l):
            r = random.random()
            if r < 0.15:
                yield random.choice(self.pos_set)
            elif r < 0.4:
                yield random.choice(self.noise_set)
            elif r < 0.6:
                yield random.choice(self.neg_set3)
            elif r < 0.8:
                try:
                    val = next(self.neg_set)
                except StopIteration:
                    self.neg_set = iter(self.neg_ds)
                    val = next(self.neg_set)
                yield val
            else:
                try:
                    val = next(self.neg_set2)
                except StopIteration:
                    self.neg_set2 = iter(self.neg_ds2)
                    val = next(self.neg_set2)
                yield val

class NoiseSet():
    def __init__(self):
        self.root_dir = "speech-data"
        self.noise = []
        for p in Path(f"{self.root_dir}/_background_noise_").iterdir():
            if not p.name.endswith(".wav"): continue
            self.noise.append(AudioSegment.from_file(p))
        for _ in range(5):
            self.noise.append(AudioSegment.silent(1000, 16000))
    
    def __len__(self):
        return len(self.noise)
    
    def __getitem__(self, idx):
        n = self.noise[idx]
        start_idx = random.randint(0, len(n)-1000)
        n = n[start_idx:start_idx+1000]
        n = augment(n, self.noise)
        audio = np.array(n.get_array_of_samples())
        audio_frames = prepare_data(audio)
        features = np.expand_dims(process_frames(audio_frames), axis=0)
        return features, 0


class PeoplesSpeech:
    def __init__(self, hf_name="clean", split="train"):
        self.hf_name = hf_name
        self.split = split

    def _generator(self, ds):
        for data in ds:
            audio = data["audio"]
            # duration = data["duration_ms"]
            text = data["text"].split()
            if audio["sampling_rate"] != 16000:
                continue
            if "stop" in text or "go" in text:
                continue
            audio_arr = audio["array"]
            if len(audio_arr) <= 16000: continue
            idx = random.randint(0, len(audio_arr) - 16000)
            audio_arr = audio_arr[idx:idx+16000]

            # random gain
            gain = random.random() * 2 + 0.5 # lin gain from 0.5, 2.5

            # break up into frames
            audio_frames = prepare_data(audio_arr)
            features = np.expand_dims(process_frames(audio_frames), axis=0)
            yield features, 0
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            shards = 1
            idx = 0
        else:
            shards = worker_info.num_workers
            idx = worker_info.id
        ds = (load_dataset("MLCommons/peoples_speech", self.hf_name, streaming=True, token=token)
              .select_columns(["audio", "text"])
        )
        ds = ds[self.split]
        ds = ds.shard(shards, idx).shuffle()
        return self._generator(ds)
        
class WavDataset(Dataset):
    def __init__(self, set_path):
        self.root_dir = "speech-data"
        with open(set_path) as f:
            self.file_paths = [line.strip() for line in f.readlines()]
        self.mfcc = None
        self.noise = []
        for p in Path(f"{self.root_dir}/_background_noise_").iterdir():
            if not p.name.endswith(".wav"): continue
            self.noise.append(AudioSegment.from_file(p))
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        if self.mfcc is None:
            self.mfcc = MFCC()
        fpath = self.file_paths[idx]
        wavpath = f"{self.root_dir}/{fpath}"
        audio = load_wav(wavpath)
        if "augmented" not in fpath:
            audio = AudioSegment(audio.tobytes(),
                                frame_rate=16000,
                                sample_width=audio.dtype.itemsize,
                                channels=1)
            audio = np.array(augment(audio, self.noise).get_array_of_samples())
        if fpath.startswith("stop"):
            label = 1
        elif fpath.startswith("go"):
            label = 2
        else:
            label = 0
        audio_frames = prepare_data(audio)
        features = np.expand_dims(process_frames(audio_frames), axis=0)
        return features, label


if __name__ == "__main__":
    # ds = CombinedDataset()
    # dsi = iter(ds)
    # total = 0
    # for i in range(1000):
        # input, label = next(dsi)
        # total += label
    # print(f"{total}/1000 positive")
    ds = (load_dataset("MLCommons/peoples_speech", "clean", streaming=True, token=token)
            .select_columns(["audio", "text"])
    )
    ds = ds["train"]
    dsi = iter(ds)
    data = next(dsi)
    aud = np.array(data["audio"]["array"])
    aud = aud * 0xffff/2
    aud = np.array(aud, dtype=np.int16)
    seg = AudioSegment(aud.tobytes(), frame_rate=16000, sample_width=2, channels=1)
    seg.export("test.wav", format="wav")
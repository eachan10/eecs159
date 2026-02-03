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
    def __init__(self, split="train", txt_path="train_pos_set.txt", txt_path_neg="train_neg_set.txt", batches=2000, batch_size=64):
        super().__init__()
        self.neg_ds = PeoplesSpeech("clean", split)
        self.neg_set = iter(self.neg_ds)
        self.neg_ds2 = PeoplesSpeech("dirty", split)
        self.neg_set2 = iter(self.neg_ds2)
        self.neg_set3 = WavDataset(txt_path_neg)
        self.len = batches * batch_size
        self.pos_set = WavDataset(txt_path)
        self.noise_set = NoiseSet()
    
    def __len__(self):
        return self.len
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self.len = self.len // worker_info.num_workers
        pos_set = self.pos_set
        noise_set = self.noise_set
        neg_set = self.neg_set
        neg_set2 = self.neg_set2
        neg_set3 = self.neg_set3
        for _ in range(self.len):
            r = random.random()
            if r < 0.2:
                yield random.choice(pos_set)
            elif r < 0.3:
                yield random.choice(noise_set)
            elif r < 0.4:
                yield random.choice(neg_set3)
            elif r < 0.7:
                try:
                    val = next(neg_set)
                except StopIteration:
                    self.neg_set = iter(self.neg_ds)
                    neg_set = self.neg_set
                    val = next(neg_set)
                yield val
            else:
                try:
                    val = next(neg_set2)
                except StopIteration:
                    self.neg_set2 = iter(self.neg_ds2)
                    neg_set2 = self.neg_set2
                    val = next(neg_set2)
                yield val

class NoiseSet():
    def __init__(self):
        self.root_dir = "speech-data"
        self.noise = []
        for p in Path(f"{self.root_dir}/_background_noise_").iterdir():
            if not p.name.endswith(".wav"): continue
            self.noise.append(AudioSegment.from_file(p))
    
    def __len__(self):
        return len(self.noise)
    
    def __getitem__(self, idx):
        n = self.noise[idx]
        start_idx = random.randint(0, len(n)-1000)
        n = n[start_idx:start_idx+1000]
        audio = np.array(n.get_array_of_samples())
        audio_frames = prepare_data(audio)
        features = np.expand_dims(process_frames(audio_frames), axis=0)
        return features, 0


class PeoplesSpeech:
    def __init__(self, hf_name="clean", split="train"):
        self.hf_name = hf_name
        self.ds = (load_dataset("MLCommons/peoples_speech", hf_name, streaming=True, token=token)
              .select_columns(["audio", "text"])
              .shuffle()
        )
        self.ds = self.ds["train"]
    
    def _generator(self, ds):
        for data in ds:
            audio = data["audio"]
            # duration = data["duration_ms"]
            text = data["text"]
            if audio["sampling_rate"] != 16000:
                continue
            if "stop" in text:
                continue
            audio_arr = audio["array"]
            idx = random.randint(0, len(audio_arr) - 16000)
            audio_arr = audio_arr[idx:idx+16000]

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
        ds = self.ds.shuffle().shard(shards, idx)
        return self._generator(ds)

# class LJSpeechDataset(Dataset):
#     REPEAT_COUNT = 20
#     def __init__(self, root_dir):
#         self.root_dir = root_dir
#         self.fnames = []
#         with open(f"{root_dir}/metadata.csv", encoding='utf-8') as f:
#             for line in f.readlines():
#                 fname, transcription, _ = line.split("|")
#                 if "stop" not in transcription:
#                     self.fnames.append(fname)
    
#     def __len__(self):
#         return len(self.fnames) * self.REPEAT_COUNT
    
#     @lru_cache(maxsize=512)
#     def get_audio_from_idx(self, idx):
#         fpath = f"{self.root_dir}/wavs/{self.fnames[idx]}.wav"
#         audio = load_wav(fpath, downsample=True)
#         return audio
    
#     def __getitem__(self, idx):
#         idx = idx % len(self.fnames)
#         audio = self.get_audio_from_idx(idx)
#         if len(audio) > 16000:
#             start_idx = random.randint(0, len(audio) - 16000)
#         else:
#             start_idx = 0
#         label = 0
#         tries = 0
#         while 1:
#             audio_frames = prepare_data(audio[start_idx:])
#             features = np.zeros((FRAMES_PER_SEC, COEF_PER_FRAME), dtype=np.float32)
#             for idx, frame in enumerate(audio_frames):
#                 # features[idx] = self.mfcc(frame)
#                 features[idx] = mfcc(frame)
#             if not np.isfinite(features).all():
#                 if tries < 3:
#                     tries += 1
#                     continue
#                 np.save("audio_frames.npy", audio_frames)
#                 np.save("features.npy", features)
#                 raise RuntimeError(f"Input has NaN/Inf from file {fpath}")
#             return features.flatten(), label
        
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
        label = 1 if fpath.startswith("stop") else 0
        audio_frames = prepare_data(audio)
        features = np.expand_dims(process_frames(audio_frames), axis=0)
        return features, label


if __name__ == "__main__":
    ds = CombinedDataset()
    dsi = iter(ds)
    total = 0
    for i in range(1000):
        input, label = next(dsi)
        total += label
    print(f"{total}/1000 positive")
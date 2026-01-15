'''
Torch model that takes in 12 MFCC coef * 400 frames = 480 input parameters
infer whether the sample contains the word stop or does not contain stop
'''

# TODO: normalize audio volume before calculating the MFCC

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm

from audio_preprocessor import *


SAMPLE_FREQ = 16000
COEF_PER_FRAME = 12
FRAME_LENGTH = 400
FRAME_STEP = 300
FRAMES_PER_SEC = len(range(0, SAMPLE_FREQ - FRAME_LENGTH, FRAME_STEP)) # 40
INPUT_SIZE = COEF_PER_FRAME * FRAMES_PER_SEC


class WavDataset(Dataset):
    def __init__(self, set_path):
        self.root_dir = "speech-data"
        with open(set_path) as f:
            self.file_paths = [line.strip() for line in f.readlines()]
        self.mfcc = None
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        if self.mfcc is None:
            self.mfcc = MFCC()
        fpath = self.file_paths[idx]
        wavpath = f"{self.root_dir}/{fpath}"
        audio = load_wav(wavpath)
        label = 1 if fpath.startswith("stop") else 0
        tries = 0
        while 1:
            audio_frames = prepare_data(audio)
            features = np.zeros((FRAMES_PER_SEC, COEF_PER_FRAME), dtype=np.float32)
            for idx, frame in enumerate(audio_frames):
                # features[idx] = self.mfcc(frame)
                features[idx] = mfcc(frame)
            # features = process_frames(audio_frames)
            if not np.isfinite(features).all():
                if tries < 3:
                    tries += 1
                    continue
                np.save("audio_frames.npy", audio_frames)
                np.save("features.npy", features)
                raise RuntimeError(f"Input has NaN/Inf from file {fpath}")
            return features.flatten(), label

BATCH_SIZE = 64
training_data_loader = DataLoader(WavDataset("training_set.txt"),
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=4,
                                  prefetch_factor=2,
                                  )
testing_data_loader = DataLoader(WavDataset("testing_set.txt"),
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=4,
                                 prefetch_factor=2,
                                 )
validation_data_loader = DataLoader(WavDataset("validation_set.txt"),
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=4,
                                    prefetch_factor=2,
                                    )

class Net(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(INPUT_SIZE, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

THRESHOLD = 0.7

def validate(nn, threshold, data_loader):
    nn.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for data in tqdm.tqdm(data_loader, desc="Validation", unit="Batches"):
            wavs, labels = data
            outputs = nn(wavs)
            predicted = torch.sigmoid(outputs[:,0]) > threshold
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    nn.train()
    print(f"Accuracy is {correct}/{total} : {correct*100/total:.2f}%")

def train(net, epochs, w):
    training_data_loader = DataLoader(WavDataset("training_set.txt"),
                                      batch_size=BATCH_SIZE,
                                      shuffle=True,
                                      num_workers=4,
                                      prefetch_factor=2,
                                      )
    weights = torch.tensor([w], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0, weight_decay=1e-4)
    running_loss_batches = 20
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} starting...")
        running_loss = 0.0
        bar = tqdm.tqdm(enumerate(training_data_loader, 0),
                        total=len(training_data_loader),
                        unit="Batches")
        for i, data in bar:
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % running_loss_batches == running_loss_batches - 1:
                bar.set_description(f'Batches: {i+1-running_loss_batches}-{i + 1} loss: {running_loss / running_loss_batches:.3f}')
                running_loss = 0.0
        # validate(net, THRESHOLD, validation_data_loader)

def main():
    LOAD = False
    EPOCHS = 5
    MODEL_LOAD_PATH = "./model.pth"
    MODEL_SAVE_PATH = "./model.pth"
    net = Net()
    if LOAD:
        net.load_state_dict(torch.load(MODEL_LOAD_PATH, weights_only=True))
    try:
        train(net, EPOCHS, 14)
    finally:
        torch.save(net.state_dict(), MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
    print("Finished Training")
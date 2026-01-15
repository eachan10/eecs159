from nn import Net, train, validate, WavDataset
import augment_data
import sort_data

import torch
from torch.utils.data import DataLoader

import shutil
import os

torch.autograd.set_detect_anomaly(True)

def main():
    os.makedirs("out", exist_ok=True)
    net = Net()

    try:
        for i in range(5):
            print("============================")
            print(f"      Dataset #{i}")
            print("============================")
            if os.path.exists("speech-data/stop-augmented"):
                shutil.rmtree("speech-data/stop-augmented")
            if os.path.exists("speech-data/neg-augmented"):
                shutil.rmtree("speech-data/neg-augmented")
            os.makedirs("speech-data/stop-augmented", exist_ok=True)
            os.makedirs("speech-data/neg-augmented", exist_ok=True)
            print("Augmenting data...")
            augment_data.main()
            print("Sorting data...")
            weight = sort_data.main()
            print("Training...")
            train(net, 2, weight)
            validation_data_loader = DataLoader(WavDataset("validation_set.txt"),
                                        batch_size=64,
                                        shuffle=True,
                                        num_workers=4,
                                        prefetch_factor=2,
                                        )
            print("Validating...")
            validate(net, 0.7, validation_data_loader)
            torch.save(net.state_dict(), f"out/model_iter{i}.pth")
    finally:
        torch.save(net.state_dict(), "out/model_final.pth")

if __name__ == "__main__":
    main()
import torch
from nn import Net, WavDataset, validate, validation_data_loader
from torch.utils.data import Dataset, DataLoader
import time

if __name__ == "__main__":
    PATH = "model-2.pth"
    net = Net()
    net.load_state_dict(torch.load(PATH, weights_only=True))

    stop_loader = DataLoader(WavDataset("stop_set.txt"),
                             batch_size=64,
                             shuffle=True,
                             num_workers=4,
                             prefetch_factor=2,
                             )
    THRESHOLD = 0.5

    time.sleep(5)
    # print("Test stop only")
    # validate(net, THRESHOLD, stop_loader)
    print("Test validation set")
    validate(net, THRESHOLD, validation_data_loader)

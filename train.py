from nn import Net, ConvNet, train_loop, validate, fuse_module, quantize
from data import CombinedDataset

import torch
from torch.utils.data import DataLoader
import torch.ao.quantization as quant

import os

torch.autograd.set_detect_anomaly(True)

QAT = True
EPOCH_LENGTH = 10000

def main():
    os.makedirs("out", exist_ok=True)
    # net = Net()
    net = ConvNet()
    if QAT:
        quant.prepare_qat(net, inplace=True)
    net.load_state_dict(torch.load("out/model_final.pth", weights_only=True))
    dataset = CombinedDataset("train",
                              "clean", "dirty",
                              "train_pos_set.txt", "train_neg_set.txt",
                              4000, 64)
    dataset_val = CombinedDataset("train",
                                  "clean_sa", "dirty_sa",
                                  "val_pos_set.txt", "val_neg_set.txt", 400, 64)
    validation_data_loader = DataLoader(dataset_val,
                                        batch_size=64,
                                        num_workers=4,
                                        prefetch_factor=4,
                                        persistent_workers=True,
                                       )
    training_data_loader = DataLoader(dataset,
                                      batch_size=64,
                                      num_workers=8,
                                      prefetch_factor=4,
                                      persistent_workers=True,
                                     )
    try:
        for i in range(10):
            print("============================")
            print(f"      EPOCH #{i}")
            print("============================")

            print("Training...")
            # iter0, iter1 are fp32 models
            # iter2... are fused but not quantized
            if i >= 9:
                net.eval() # disable dropout for QAT
                net.apply(quant.disable_observer)
            else:
                net.train()
                # net.apply(quant.enable_observer)
            if i >= 5:
                lr = 0.0001 * (0.95**i)
            else:
                lr = 0.0002 * (0.95**i)
            train_loop(net, lr=lr, w=1, loader=training_data_loader)
            print("Validating...")
            net.eval()
            validate([net], [0.3,0.4,0.5,0.6,0.7,0.8], validation_data_loader)
            net.train()
            torch.save(net.state_dict(), f"out/model_iter{i}.pth")
    finally:
        # if QAT:
            # fuse_module(net)
            # quantize(net)
        torch.save(net.state_dict(), "out/model_final.pth")

if __name__ == "__main__":
    main()
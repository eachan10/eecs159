import torch
from nn import ConvNet, Net, ConvNetStopGo, validate, validate_stop_go, fuse_module, quantize
from torch.utils.data import DataLoader
from torch.ao import quantization as quant
import time
from test import MathNet
from data import WavDataset, CombinedDataset

QAT = True

if __name__ == "__main__":
    # PATH = "model_after_prepare_qat.pth"
    # PATH = "model_after_prepare_qat.pth"
    PATH = "out/model_iter1.pth"
    nets = []
    for i in range(10):
        path = f"out-conv-stop-go/model_iter{i}.pth"
        net = ConvNetStopGo()
        # first are fp32 models
        # next are QAT models
        if QAT:
            quant.prepare_qat(net, inplace=True)
        net.load_state_dict(torch.load(path, weights_only=True))
        # net.load_state_dict(torch.load(f"out/model_iter{i}.pth", weights_only=True))
        if QAT:
            # fuse_module(net)
            quantize(net)
        nets.append(net)
        net.eval()
    for i in range(10):
        path = f"out-conv-stop-go-float/model_iter{i}.pth"
        net = ConvNetStopGo()
        # first are fp32 models
        # next are QAT models
        net.load_state_dict(torch.load(path, weights_only=True))
        # net.load_state_dict(torch.load(f"out/model_iter{i}.pth", weights_only=True))
        nets.append(net)
        net.eval()

#    math_net = MathNet(net)

    stop_loader = DataLoader(WavDataset("val_pos_set.txt"),
                             batch_size=64,
                             shuffle=True,
                             num_workers=4,
                             prefetch_factor=2,
                             )
    validation_data_loader = DataLoader(WavDataset("val_neg_set.txt"),
                             batch_size=64,
                             shuffle=True,
                             num_workers=4,
                             prefetch_factor=2,
                             )
    dataset_val = CombinedDataset("train",
                                  "clean_sa", "dirty_sa",
                                  "val_pos_set.txt", "val_neg_set.txt", 400, 64)
    validation_data_loader = DataLoader(dataset_val,
                                        batch_size=64,
                                        num_workers=4,
                                        prefetch_factor=4,
                                        persistent_workers=True,
                                       )
    # THRESHOLD = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8]

    time.sleep(5)
    # print("=====================================")
    # print("       Test quantized model")
    # print("=====================================")
    # print("Test stop only")
    #validate(net, THRESHOLD, stop_loader)
    # print("Test validation set")
    #validate(net, THRESHOLD, validation_data_loader)
    print("Nets 0-9: int8 quantized")
    print("Nets 10-19: float")
    validate_stop_go(nets, validation_data_loader)
    # validate(nets, THRESHOLD, stop_loader)
    # print("Test neg set")
    # validate(nets, THRESHOLD, validation_data_loader)

import torch
from nn import Net, validate, fuse_module, quantize
from torch.utils.data import DataLoader
from torch.ao import quantization as quant
import time
from test import MathNet
from data import WavDataset

if __name__ == "__main__":
    PATH = "model_after_prepare_qat.pth"
    # PATH = "model_after_prepare_qat.pth"
    # PATH = "out/model_final.pth"
    net = Net()
    nets = []
    for i in range(1):
        net = Net()
        # first are fp32 models
        # next are QAT models
        quant.prepare_qat(net, inplace=True)
        net.load_state_dict(torch.load(PATH, weights_only=True))
        # net.load_state_dict(torch.load(f"out/model_iter{i}.pth", weights_only=True))
        fuse_module(net)
        quantize(net)
        nets.append(MathNet(net))
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
    THRESHOLD = [0.5, 0.6, 0.7, 0.75, 0.8]

    time.sleep(5)
    # print("=====================================")
    # print("       Test quantized model")
    # print("=====================================")
    # print("Test stop only")
    #validate(net, THRESHOLD, stop_loader)
    # print("Test validation set")
    #validate(net, THRESHOLD, validation_data_loader)
    print("=====================================")
    print("           Test math model")
    print("=====================================")
    print("Nets 0-9: fp32")
    print("Nets 10-19: int8 quantized weights int32 accumulators")
    print("Test pos only")
    validate(nets, THRESHOLD, stop_loader)
    print("Test neg set")
    validate(nets, THRESHOLD, validation_data_loader)

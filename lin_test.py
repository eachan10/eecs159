from test import *
from nn import ConvNetStopGo
import torch.quantization as quant
import numpy as np
from data import WavDataset
import torch.nn.functional as F
import torch
import re

def check(golden):
    vals = []
    count = 0
    with open("src/run.txt", encoding='utf-16') as f:
        for l in f.readlines():
            l = re.sub(r"[^\d+-]", " ", l)
            if not l.startswith("Acc") and not l.startswith("Out"):
                for n in [int(w.strip()) for w in l.split()]:
                    vals.append(n)
    idx = 0
    ch = 0
    row = 0
    col = 0
    for v, g in zip(vals, golden.flatten()):
        if abs(v-g) > 1:
            print(v, g, idx, ch, row, col)
            count += 1
            if count > 10: return
        idx += 1
        col += 1
        if col == golden.size(2):
            col = 0
            row += 1
        if row == golden.size(1):
            row = 0
            ch += 1
    print (count)
    return vals

def conv2d(inp, weight):
    # padding = 1
    # inp is shape (ch, row, col)
    out = torch.zeros((weight.size(0), inp.size(1), inp.size(2))).to(torch.int32)
    for out_channel in range(weight.size(0)):
        for out_row in range(inp.size(1)):
            for out_col in range(inp.size(2)):
                acc = torch.tensor(0).to(torch.int32)
                for in_channel in range(inp.size(0)):
                    for ker_row in range(weight.size(2)):
                        for ker_col in range(weight.size(3)):
                            in_row = out_row + ker_row - 1
                            in_col = out_col + ker_col - 1

                            if (in_row >= 0 and in_row < inp.size(1) and in_col >= 0 and in_col < inp.size(2)):
                                acc += weight[out_channel, in_channel, ker_row, ker_col] * inp[in_channel, in_row, in_col]
                out[out_channel, out_row, out_col] = acc
    return out
                


if __name__ == "__main__":
    net = ConvNetStopGo()
    quant.prepare_qat(net, inplace=True)
    net.load_state_dict(torch.load("out-conv-stop-go/model_iter9.pth"))
    fuse_module(net)
    quantize(net)
    net.eval()

    pos_set = WavDataset("val_pos_set.txt")
    neg_set = WavDataset("val_neg_set.txt")
    x = torch.randn((100, 1, 52, 12))
    for i in range(50):
        data, label = pos_set[i]
        data = torch.tensor(data)
        x[i] = data
    for i in range(50, 100):
        data, label = neg_set[i]
        data = torch.tensor(data)
        x[i] = data

    xq = net.quant(x)
    x1 = net.pool1(net.relu1(net.bn1(net.conv1(xq))))
    x2 = net.pool2(net.relu2(net.bn2(net.conv2(x1))))
    x3 = net.pool3(net.relu3(net.bn3(net.conv3(x2))))
    x4 = net.flatten(x3)
    golden = net.fc1(x4)

    conv1_params = layer_params(net.quant.scale, net.conv1)
    conv2_params = layer_params(net.conv1.scale, net.conv2)
    conv3_params = layer_params(net.conv2.scale, net.conv3)
    fc1_params = layer_params(torch.tensor(x4.q_scale()), net.fc1)

    with open("src/model_dump.h", "w") as f:
        f.write(f"int32_t input_zero_point = {net.quant.zero_point.item()};\n")
        f.write(dump_params_array(conv1_params, "conv1", transpose=True))
        f.write("\n")
        f.write(dump_params_array(conv2_params, "conv2", transpose=True))
        f.write("\n")
        f.write(dump_params_array(conv3_params, "conv3", transpose=True))
        f.write("\n")
        f.write(dump_params_array(fc1_params, "fc1"))
        f.write("\n")
        pool_scale = round(1/(net.pool3.kernel_size[0]*net.pool3.kernel_size[1])*(1<<25))
        f.write(f"int32_t avgpool_acc_scale = {pool_scale};\n")
        f.write("int32_t avgpool_shift = 25;\n")

    # with open("src/test_vec.h", "w") as f:
    #     # test_x = x4.int_repr().to(torch.int32) - x4.q_zero_point()
    #     test_x = xq.int_repr().to(torch.int32) - xq.q_zero_point()
    #     f.write(f"int32_t x[{test_x.size(0)}][{test_x.numel()//test_x.size(0)}] = {{\n")
    #     for row in test_x:
    #         count = 0
    #         f.write("  {\n")
    #         for col in row.flatten():
    #             if count == 0:
    #                 f.write("    ")
    #             f.write(f"{col.item():5}, ")
    #             if count == 15:
    #                 f.write("\n")
    #                 count = 0
    #             else:
    #                 count += 1
    #         f.write("\n  },\n")
    #     f.write("\n};\n")
    #     test_y = golden.int_repr().to(torch.int32) - net.fc1.zero_point
    #     f.write(f"int32_t y[{test_y.numel()}] = {{\n    ")
    #     for col in test_y.flatten():
    #         if count == 0:
    #             f.write("    ")
    #         f.write(f"{col.item():5}, ")
    #         if count == 15:
    #             f.write("\n")
    #             count = 0
    #         else:
    #             count += 1
    #     f.write("};\n")

from test import *
from nn import ConvNet
import torch.quantization as quant
import numpy as np
from data import WavDataset

if __name__ == "__main__":
    net = ConvNet()
    quant.prepare_qat(net, inplace=True)
    net.load_state_dict(torch.load("out-cnn-32-64-128-fc1/model_iter9.pth"))
    quantize(net)
    net.eval()

    pos_set = WavDataset("val_pos_set.txt")
    neg_set = WavDataset("val_neg_set.txt")
    x = torch.empty((10, 1, 52, 12))
    for i in range(5):
        data, label = pos_set[i]
        data = torch.tensor(data)
        x[i] = data
    for i in range(5, 10):
        data, label = neg_set[i]
        data = torch.tensor(data)
        x[i] = data

    xq = net.quant(x)
    x1 = net.pool1(net.relu1(net.bn1(net.conv1(xq))))
    x2 = net.pool2(net.relu2(net.bn2(net.conv2(x1))))
    x3 = net.pool3(net.relu3(net.bn3(net.conv3(x2))))
    x4 = net.flatten(x3)
    golden = net.fc1(x4)

    fc1_params = layer_params(torch.tensor(x4.q_scale()), net.fc1)
    fc1_params["weight"].detach().numpy().astype(np.int32).tofile("src/weights.bin")

    print((golden.int_repr().to(torch.int32) - golden.q_zero_point()).flatten())
    
    def test(idx):
        return quantized_linear_c(fc1_params, x4[idx:idx+1].int_repr() - x4.q_zero_point())
    test_out =[]
    for i in range(10):
        test_out.append(test(i).item())
    print(test_out)
    with open("src/model_lin.h", "w") as f:
        f.write(dump_params_array(fc1_params, "fc1"))

    with open("src/lin_test.h", "w") as f:
        test_x = x4.int_repr().to(torch.int32) - x4.q_zero_point()
        f.write(f"int32_t x[{test_x.size(0)}][{xq.numel()//xq.size(0)}] = {{\n")
        for row in test_x:
            count = 0
            f.write("  {\n")
            for col in row.flatten():
                if count == 0:
                    f.write("    ")
                f.write(f"{col.item():5}, ")
                if count == 15:
                    f.write("\n")
                    count = 0
                else:
                    count += 1
            f.write("\n  },\n")
        f.write("\n};\n")
        test_y = golden.int_repr().to(torch.int32) - net.fc1.zero_point
        f.write(f"int32_t y[{test_y.numel()}] = {{")
        for col in test_y:
            f.write(f"{col.item():6}, ")
        f.write("};\n")

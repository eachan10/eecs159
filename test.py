from nn import Net, fuse_module, quantize
import torch
import torch.ao.quantization as quant


def quantize_input(x, scale, zero_point):
    return torch.round(x / scale) + zero_point

def extract_qlin_params(lin: torch.ao.nn.quantized.Linear):
    """Get necessary parameters from quantized linear layer
    
    Get int8 weights, fp32e bias,
    weight scale, weight zero point,
    output scale, output zero point"""
    weight_int8 = lin.weight().int_repr()
    weight_scale = torch.tensor(lin.weight().q_scale(), dtype=torch.float32)
    weight_zero_point = lin.weight().q_zero_point()
    output_scale = torch.tensor(lin.scale, dtype=torch.float32)
    output_zero_point = lin.zero_point
    biasfp32 = lin.bias()
    return (weight_int8, biasfp32, weight_scale, weight_zero_point, output_scale, output_zero_point)

def quantized_linear(x_uint8, x_scale, x_zero_point,
                     weight_int8, weight_scale, weight_zero_point,
                     bias_fp32,
                     output_scale, output_zero_point):
    x_int32 = x_uint8.to(torch.int32) - x_zero_point
    weight_int32 = weight_int8.to(torch.int32) - weight_zero_point
    acc_int32 = torch.mm(x_int32, weight_int32.T).to(torch.int32)
    bias_scale = x_scale * weight_scale
    bias_int32 = (bias_fp32 / bias_scale).to(torch.int32)
    acc_int32 = acc_int32 + bias_int32
    USE_FLOAT=False
    if USE_FLOAT:
        acc_scale = torch.round(1 / (x_scale * weight_scale / output_scale)).to(torch.int32)
        acc_int32 = torch.round(acc_int32 / acc_scale) + output_zero_point  # float div
    else:
        SHIFT = 22
        acc_int32 *= acc_scale
        acc_int32 += (1 << (SHIFT - 1))
        acc_int32 >>= SHIFT
        acc_int32 += output_zero_point
    out_uint8 = torch.clamp(acc_int32, 0, 255).to(torch.uint8)
    return out_uint8, output_scale, output_zero_point

def fake_forward(lin: torch.ao.nn.quantized.Linear, x: torch.Tensor):
    x_scale = x.q_scale()
    x_zero_point = x.q_zero_point()
    x_uint8 = x.int_repr()
    w_int8, bias_fp32, w_scale, w_zero_point, out_scale, out_zero_point = extract_qlin_params(lin)
    q_out = quantized_linear(x_uint8,
                             x_scale,
                             x_zero_point,
                             w_int8,
                             w_scale,
                             w_zero_point,
                             bias_fp32,
                             out_scale,
                             out_zero_point)
    # out_uint8, out_scale, out_zero_point = q_out
    return q_out

class MathNet:
    def __init__(self, net: Net):
        self.in_scale = net.quant.scale.to(torch.float32)
        self.in_zero_point = net.quant.zero_point.to(torch.int32)
        self.fc1_params = extract_qlin_params(net.fc1)
        self.fc2_params = extract_qlin_params(net.fc2)
        self.fc3_params = extract_qlin_params(net.fc3)
        self.fc4_params = extract_qlin_params(net.fc4)
    
    def eval(self):
        ...
    def train(self):
        ...
    def __call__(self, x):
        return self.forward(x)
    
    def quantize(self, x):
        return (torch.round((x/self.in_scale)) + self.in_zero_point).to(torch.uint8)
    
    def forward(self, x):
        # quantize input
        uint8_x = self.quantize(x)
        x_scale = self.in_scale
        x_zero_point = self.in_zero_point

        # fully connected layer 1
        w_int8, bias_fp32, w_scale, w_zero_point, out_scale, out_zero_point = self.fc1_params
        uint8_x, x_scale, x_zero_point = quantized_linear(uint8_x, x_scale, x_zero_point,
                         w_int8, w_scale, w_zero_point,
                         bias_fp32,
                         out_scale, out_zero_point)
        uint8_x = torch.clamp(uint8_x, min=x_zero_point) # ReLU
        out1 = uint8_x
        # fully connected layer 2
        w_int8, bias_fp32, w_scale, w_zero_point, out_scale, out_zero_point = self.fc2_params
        uint8_x, x_scale, x_zero_point = quantized_linear(uint8_x, x_scale, x_zero_point,
                         w_int8, w_scale, w_zero_point,
                         bias_fp32,
                         out_scale, out_zero_point)
        uint8_x = torch.clamp(uint8_x, min=x_zero_point) # ReLU
        out2 = uint8_x
        # fully connected layer 3
        w_int8, bias_fp32, w_scale, w_zero_point, out_scale, out_zero_point = self.fc3_params
        uint8_x, x_scale, x_zero_point = quantized_linear(uint8_x, x_scale, x_zero_point,
                         w_int8, w_scale, w_zero_point,
                         bias_fp32,
                         out_scale, out_zero_point)
        uint8_x = torch.clamp(uint8_x, min=x_zero_point) # ReLU
        out3 = uint8_x
        # fully connected layer 4
        w_int8, bias_fp32, w_scale, w_zero_point, out_scale, out_zero_point = self.fc4_params
        uint8_x, x_scale, x_zero_point = quantized_linear(uint8_x, x_scale, x_zero_point,
                         w_int8, w_scale, w_zero_point,
                         bias_fp32,
                         out_scale, out_zero_point)
        out4 = uint8_x
        # dequantize output
        # return out1, out2, out3, out4
        # return out4
        fp32_out = (uint8_x.to(torch.int32) - x_zero_point).to(torch.float32) * x_scale
        return fp32_out


if __name__ == "__main__":
    net = Net()
    quant.prepare_qat(net, inplace=True)
    net.load_state_dict(torch.load("model_after_prepare_qat.pth"))
    fuse_module(net)
    quantize(net)
    net.eval()

    x = torch.randn(10,624) * 10 - 50

    math_net = MathNet(net)
    golden = net.forward(x)

    x1 = net.relu1(net.fc1(net.quant(x)))
    x2 = net.relu2(net.fc2(x1))
    x3 = net.relu3(net.fc3(x2))
    x4 = net.fc4(x3)
    x1 = x1.int_repr()
    x2 = x2.int_repr()
    x3 = x3.int_repr()
    # out1, out2, out3, out4 = math_net.forward(x)
    out = math_net.forward(x)
    # print(f"Golden: {x4.int_repr()}")
    print(f"Golden: {golden}")
    print(f"Out:    {out}")

    # t= torch.randn(5,64)
    # t = net.quant(t)
    # print(net.fc4(t).int_repr())
    # print(fake_forward(net.fc4, t))

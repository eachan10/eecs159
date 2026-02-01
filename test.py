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
    bias_int32 = torch.round(bias_fp32 / bias_scale).to(torch.int32)
    acc_int32 = acc_int32 + bias_int32
    USE_FLOAT=False
    if USE_FLOAT:
        acc_scale = torch.round(1 / (x_scale * weight_scale / output_scale)).to(torch.int32)
        acc_int32 = torch.round(acc_int32 / acc_scale) + output_zero_point  # float div
    else:
        SHIFT = 22
        acc_scale = torch.round((1 << SHIFT) * (x_scale * weight_scale / output_scale)).to(torch.int32)
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
        self.fc1, self.fc2, self.fc3, self.fc4 = dump_params(net)
    
    def eval(self):
        ...
    def train(self):
        ...
    def __call__(self, x):
        return self.forward(x)
    
    def quantize(self, x):
        return torch.clamp((torch.round((x/self.in_scale)) + self.in_zero_point), 0, 255).to(torch.uint8)
    
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
    
    def forward2(self, x):
        x = quantize_x(x, self.in_scale, self.in_zero_point)
        # FC1
        x = quantized_linear_c(self.fc1, x)
        # ReLU
        x = torch.clamp(x, min=0)
        # FC2
        x = quantized_linear_c(self.fc2, x)
        # ReLU
        x = torch.clamp(x, min=0)
        # FC3
        x = quantized_linear_c(self.fc3, x)
        # ReLU
        x = torch.clamp(x, min=0)
        # FC4
        x = quantized_linear_c(self.fc4, x)
        # dequantize
        x = x
        return x
    
    def dequantize(self, x):
        return x.to(torch.float32) * self.fc4["output_scale"]

def layer_params(x_scale, lin):
    SHIFT = 22
    curr = {}
    w = lin.weight()
    b_scale = x_scale * w.q_scale()
    b = torch.round(lin.bias() / b_scale).to(torch.int32)
    acc_scale = x_scale*w.q_scale()/torch.tensor(lin.scale)
    curr["weight"]            = w.int_repr()
    curr["bias"]              = b
    curr["output_scale"]      = torch.tensor(lin.scale)
    curr["output_zero_point"] = torch.tensor(lin.zero_point)
    curr["acc_scale"]         = torch.round((1<<SHIFT)*acc_scale).to(torch.int32)
    curr["shift"]             = SHIFT
    return curr


def dump_params(net: Net):
    fc1 = layer_params(net.quant.scale.to(torch.float32), net.fc1)
    fc2 = layer_params(fc1["output_scale"], net.fc2)
    fc3 = layer_params(fc2["output_scale"], net.fc3)
    fc4 = layer_params(fc3["output_scale"], net.fc4)
    return fc1, fc2, fc3, fc4

def dump_params_array(params, prefix):
    all_str = []
    # weights 2d matrix
    w : torch.Tensor= params["weight"]
    w_str = f"int16_t {prefix}_weights[{w.size(0)}*{w.size(1)}] = {{\n"
    for idx, row in enumerate(w):
        count = 0
        # w_str += "  {\n"
        w_str += f"//////// Row {idx}  /////////\n"
        for col in row:
            if count == 0:
                w_str += "    "
            w_str += f"{col.item():5},"
            if count == 15:
                w_str += "\n"
                count = 0
            else:
                count += 1
        # w_str += "\n  },\n"
    w_str += "\n};\n"
    all_str.append(w_str)
    # bias 1d array
    b = params["bias"]
    b_str = f"int32_t {prefix}_bias[{b.size(0)}] = {{\n"
    count = 0
    for col in b:
        if count == 0:
            b_str += "  "
        b_str += f"{col.item():6},"
        if count == 15:
            b_str += "\n"
            count = 0
        else:
            count += 1
    b_str += "\n};\n"
    all_str.append(b_str)

    # output_scale int32
    # out_scale = params["output_scale"]
    # if type(out_scale) is torch.Tensor:
        # out_scale = out_scale.item()
    # os_str = f"int32_t {prefix}_output_scale = {out_scale};"
    # all_str.append(os_str)

    # output_zero_point int32
    out_zero_point = params["output_zero_point"]
    if type(out_zero_point) is torch.Tensor:
        out_zero_point = out_zero_point.item()
    ozp_str = f"int32_t {prefix}_output_zero_point = {out_zero_point};"
    all_str.append(ozp_str)
    # accumulator scale int32
    acc_scale = params["acc_scale"]
    if type(acc_scale) is torch.Tensor:
        acc_scale = acc_scale.item()
    as_str = f"int32_t {prefix}_acc_scale = {acc_scale};"
    all_str.append(as_str)
    # shfit int32?
    shift = params["shift"]
    s_str = f"int32_t {prefix}_shift = {shift};"
    all_str.append(s_str)
    return "\n".join(all_str)

def quantize_x(x, scale, zero_point):
    return torch.clamp(torch.round(x / scale), -zero_point, 255-zero_point).to(torch.int16)

def quantized_linear_c(lin: dict, x: torch.tensor):
    # I can ommit the x and output zero points add/sub
    # since I will keep things at int16
    x_int32 = x.to(torch.int32)
    w_int32 = lin["weight"].to(torch.int32)
    acc_int32 = torch.mm(x_int32, w_int32.T)
    acc_int32 += lin["bias"]
    acc_int32 *= lin["acc_scale"]
    acc_int32 += (1 << (lin["shift"]-1))  # rounding
    acc_int32 >>= lin["shift"]
    acc_int32 = torch.clamp(acc_int32, -lin["output_zero_point"], 255-lin["output_zero_point"]).to(torch.int16)
    return acc_int32

if __name__ == "__main__":
    net = Net()
    quant.prepare_qat(net, inplace=True)
    net.load_state_dict(torch.load("model_after_prepare_qat.pth"))
    fuse_module(net)
    quantize(net)
    net.eval()

    x = torch.randn(10,624) * 50

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
    out2 = math_net.forward2(x)
    # print(f"Golden: {x4.int_repr()}")
    print(f"Golden: {golden}")
    print(f"Out:    {out}")
    print(f"Out2:   {out2}")
    print(f"Out dq: {math_net.dequantize(out2)}")

    # t= torch.randn(5,64)
    # t = net.quant(t)
    # print(net.fc4(t).int_repr())
    # print(fake_forward(net.fc4, t))

    # Dump model params as C decl
    if True:
        with open("src/model_dump.h", "w") as f:
            f.write(dump_params_array(math_net.fc1, "fc1"))
            f.write("\n")
            f.write(dump_params_array(math_net.fc2, "fc2"))
            f.write("\n")
            f.write(dump_params_array(math_net.fc3, "fc3"))
            f.write("\n")
            f.write(dump_params_array(math_net.fc4, "fc4"))
            f.write("\n")

    # create test vectors
    if False:
        xq = quantize_x(x, math_net.in_scale, math_net.in_zero_point)
        x_str = f"int16_t x[{xq.size(0)}][{xq.size(1)}] = {{\n"
        for row in xq:
            count = 0
            x_str += "  {\n"
            for col in row:
                if count == 0:
                    x_str += "    "
                x_str += f"{col.item():5},"
                if count == 15:
                    x_str += "\n"
                    count = 0
                else:
                    count += 1
            x_str += "\n  },\n"
        x_str += "\n};\n"
        y = out2
        y_str = f"int16_t y[{y.size(0)}] = {{"
        for col in y:
            if count == 0:
                y_str += "  "
            y_str += f"{col.item():6},"
            if count == 15:
                y_str += "\n"
                count = 0
            else:
                count += 1
        y_str += "\n};\n"
        with open("src/test_vec.h", "w") as f:
            f.write(x_str)
            f.write(y_str)
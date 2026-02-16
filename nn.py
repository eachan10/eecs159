'''
Torch model that takes in 12 MFCC coef * 400 frames = 480 input parameters
infer whether the sample contains the word stop or does not contain stop
'''

# TODO: normalize audio volume before calculating the MFCC

import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as quant
import tqdm


from audio_preprocessor import *
from data import INPUT_SIZE

BATCH_SIZE = 64

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.qconfig = quant.default_qat_qconfig
        # self.qconfig = quant.QConfig(
        #     activation=quant.FakeQuantize.with_args(
        #         observer=quant.MinMaxObserver,
        #         qscheme=torch.per_tensor_symmetric,
        #         dtype=torch.int8
        #     ),
        #     weight=quant.FakeQuantize.with_args(
        #         observer=quant.MinMaxObserver,
        #         qscheme=torch.per_tensor_symmetric,
        #         dtype=torch.int8
        #     )
        # )
        self.qconfig = quant.get_default_qat_qconfig("qnnpack")
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()
        self.fc1 = nn.Linear(INPUT_SIZE, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64, 1)
    def forward(self, x):
        x = x.flatten()
        x = self.quant(x)
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        x = self.drop3(self.relu3(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        x = self.dequant(x)
        return x

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.qconfig = quant.get_default_qat_qconfig("qnnpack")
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()
        self.relu = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        # conv

        # input (batch, 1, 52, 12)
        # output (batch, 32, 52, 12)
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # input (batch, 32, 52, 12)
        # output (batch, 32, 26, 12)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))

        # input (batch, 32, 26, 12)
        # output (batch, 64, 26, 12)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # input (batch, 64, 26, 12)
        # output (batch, 64, 13, 6)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        # input (batch, 64, 13, 6)
        # output (batch, 64, 13, 6)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        # input (batch, 128, 13, 6)
        # output (batch, 128, 1, 1)
        self.pool3 = nn.AvgPool2d((13,6))
        # classiffier
        self.flatten = nn.Flatten() # -> 32 * 13 * 6
        self.drop1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.dequant(x)
        return x

class DSConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.qconfig = quant.get_default_qat_qconfig("qnnpack")
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()
        self.relu = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        # conv

        # input (batch, 1, 52, 12)
        # output (batch, 32, 52, 12)
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=16,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # input (batch, 32, 52, 12)
        # output (batch, 32, 26, 12)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))

        # input (batch, 32, 26, 12)
        # output (batch, 64, 26, 12)
        self.conv2a = nn.Conv2d(in_channels=16,
                               out_channels=16,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=16,
                               bias=False)
        self.conv2b = nn.Conv2d(in_channels=16,
                                out_channels=32,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        # input (batch, 64, 26, 12)
        # output (batch, 64, 13, 6)
        self.pool2 = nn.AvgPool2d(kernel_size=(13, 6))
        # self.pool2 = nn.MaxPool2d(kernel_size=2)

        # self.conv3a = nn.Conv2d(in_channels=32,
        #                         out_channels=32,
        #                         kernel_size=3,
        #                         stride=1,
        #                         padding=1,
        #                         groups=32,
        #                         bias=False)
        # self.conv3b = nn.Conv2d(in_channels=32,
        #                         out_channels=64,
        #                         kernel_size=1,
        #                         stride=1,
        #                         padding=1,
        #                         bias=False)
        # input (batch, 64, 13, 6)
        # output (batch, 64, 13, 6)
        # self.bn3 = nn.BatchNorm2d(64)

        # input (batch, 128, 13, 6)
        # output (batch, 128, 1, 1)
        # self.pool3 = nn.AvgPool2d((13,6))

        # avg pooled (64x26x12) to (64x2x2)
        # classiffier
        self.flatten = nn.Flatten() # -> 32x2x2
        self.drop1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # x = self.conv3a(x)
        # x = self.conv3b(x)
        # x = self.bn3(x)
        # x = self.relu3(x)
        # x = self.pool3(x)
        x = self.flatten(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.dequant(x)
        return x


class ConvNetStopGo(nn.Module):
    def __init__(self):
        super().__init__()
        self.qconfig = quant.get_default_qat_qconfig("qnnpack")
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()
        self.relu = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        # conv

        # input (batch, 1, 52, 12)
        # output (batch, 32, 52, 12)
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # input (batch, 32, 52, 12)
        # output (batch, 32, 26, 12)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))

        # input (batch, 32, 26, 12)
        # output (batch, 64, 26, 12)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # input (batch, 64, 26, 12)
        # output (batch, 64, 13, 6)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        # input (batch, 128, 13, 6)
        # output (batch, 128, 1, 1)
        self.pool3 = nn.AvgPool2d((13,6))
        # classiffier
        self.flatten = nn.Flatten() # -> 32 * 13 * 6
        self.drop1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128, 3)
        # self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.dequant(x)
        return x


THRESHOLD = 0.7

def validate_stop_go(nets, data_loader):
    for n in nets: n.eval()
    # array of correct count for each net for each threshold
    correct = np.zeros(len(nets), dtype=np.int32)
    correct_pos = np.zeros(len(nets), dtype=np.int32)
    correct_pos2 = np.zeros(len(nets), dtype=np.int32)
    correct_neg = np.zeros(len(nets), dtype=np.int32)
    losses = np.zeros(len(nets))
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        total = 0
        total_pos = 0
        total_pos2 = 0
        total_neg = 0
        batches = 0
        for data in tqdm.tqdm(data_loader, desc="Validation", unit="Batches"):
            wavs, labels = data
            for idx, n in enumerate(nets):
                outputs = n(wavs)
                y = labels
                loss = criterion(outputs, y)
                losses[idx] += loss

                prob = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(prob, dim=1)
                correct[idx] += (predicted == labels).sum().item()
                correct_neg[idx] += ((predicted == 0) & (labels == 0)).sum().item()
                correct_pos[idx] += ((predicted == 1) & (labels == 1)).sum().item()
                correct_pos2[idx] += ((predicted == 2) & (labels == 2)).sum().item()
            total += len(labels)
            total_neg += (labels == 0).sum().item()
            total_pos += (labels == 1).sum().item()
            total_pos2 += (labels == 2).sum().item()
            batches += 1
    for n in nets: n.train()
    for idx in range(len(nets)):
        print(f"Model #{idx}")
        print(f"Loss: {losses[idx]/batches}")
        acc = correct[idx]/total
        pos_acc = correct_pos[idx]/total_pos
        pos_acc2 = correct_pos2[idx]/total_pos
        neg_acc = correct_neg[idx]/total_neg
        print(f""
                f"Acc: {correct[idx]}/{total} : {acc*100:.2f}% "
                f"Stop: {correct_pos[idx]}/{total_pos} : {pos_acc*100:.2f}% "
                f"Go: {correct_pos2[idx]}/{total_pos} : {pos_acc2*100:.2f}% "
                f"Neg: {correct_neg[idx]}/{total_neg} : {neg_acc*100:.2f}% ")

def validate(nets, threshold, data_loader):
    for n in nets: n.eval()
    # array of correct count for each net for each threshold
    correct = np.zeros((len(nets), len(threshold)), dtype=np.int32)
    correct_pos = np.zeros((len(nets), len(threshold)), dtype=np.int32)
    correct_neg = np.zeros((len(nets), len(threshold)), dtype=np.int32)
    losses = np.zeros(len(nets))
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        total = 0
        total_pos = 0
        total_neg = 0
        batches = 0
        for data in tqdm.tqdm(data_loader, desc="Validation", unit="Batches"):
            wavs, labels = data
            for idx, n in enumerate(nets):
                outputs = n(wavs)
                y = labels.float().unsqueeze(1)
                soft_labels = y * (1 - 0.05) + (1 - y) * 0.05
                loss = criterion(outputs, y)
                losses[idx] += loss
                sig = torch.sigmoid(outputs[:,0])
                for i, t in enumerate(threshold):
                    predicted = sig > t
                    correct[idx, i] += (predicted == labels).sum().item()
                    correct_pos[idx, i] += ((predicted == 1) & (labels == 1)).sum().item()
                    correct_neg[idx, i] += ((predicted == 0) & (labels == 0)).sum().item()
            total += len(labels)
            total_pos += (labels == 1).sum().item()
            total_neg += (labels == 0).sum().item()
            batches += 1
    for n in nets: n.train()
    for idx in range(len(nets)):
        print(f"Model #{idx}")
        print(f"Loss: {losses[idx]/batches}")
        for c, cpos, cneg, t in zip(correct[idx],correct_pos[idx],correct_neg[idx],threshold):
            acc = c/total
            pos_acc = cpos/total_pos
            neg_acc = cneg/total_neg
            print(f"Threshold: {t} "
                  f"Acc: {c}/{total} : {acc*100:.2f}% "
                  f"Pos: {cpos}/{total_pos} : {pos_acc*100:.2f}% "
                  f"Neg: {cneg}/{total_neg} : {neg_acc*100:.2f}% ")

def train_loop(net, lr, w, loader):
    weights = torch.tensor([w], dtype=torch.float32)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    criterion = nn.CrossEntropyLoss()
    running_loss_batches = 20
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    running_loss = 0.0
    bar = tqdm.tqdm(enumerate(loader, 0),
                    total=len(loader),
                    unit="Batches")
    for i, data in bar:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        y = labels.float().unsqueeze(1)
        # soft_labels = y * (1 - 0.05) + (1 - y) * 0.05
        # loss = criterion(outputs, soft_labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % running_loss_batches == running_loss_batches - 1:
            bar.set_description(f'Batches: {i+1-running_loss_batches}-{i + 1} loss: {running_loss / running_loss_batches:.3f}')
            running_loss = 0.0

def fuse_module(net):
    net.eval()
    # net = quant.fuse_modules(net, [['fc1','bn1'],
    #                                ['fc2','bn2'],
    #                                ['fc3','bn3']],
    #                               fuser_func=torch.nn.utils.fuse_linear_bn_eval)
    # net = quant.fuse_modules(net, [['fc1','bn1','relu1'],
                                #    ['fc2','bn2','relu2'],
                                #    ['fc3','bn3','relu3']])
    if type(net) is Net:
        net.fc1 = torch.nn.utils.fuse_linear_bn_eval(net.fc1, net.bn1)
        net.bn1 = torch.nn.Identity()
        net.fc2 = torch.nn.utils.fuse_linear_bn_eval(net.fc2, net.bn2)
        net.bn2 = torch.nn.Identity()
        net.fc3 = torch.nn.utils.fuse_linear_bn_eval(net.fc3, net.bn3)
        net.bn3 = torch.nn.Identity()
    elif type(net) is DSConvNet:
        modules_to_fuse = [ ['conv1', 'bn1'], ['conv2b', 'bn2']]
        net.conv1 = torch.nn.utils.fuse_conv_bn_eval(net.conv1, net.bn1)
        net.bn1 = torch.nn.Identity()
        net.conv2b = torch.nn.utils.fuse_conv_bn_eval(net.conv2b, net.bn2)
        net.bn2 = torch.nn.Identity()
    elif type(net) is ConvNet:
        net.conv1 = torch.nn.utils.fuse_conv_bn_eval(net.conv1, net.bn1)
        net.bn1 = torch.nn.Identity()
        net.conv2 = torch.nn.utils.fuse_conv_bn_eval(net.conv2, net.bn2)
        net.bn2 = torch.nn.Identity()
        net.conv3 = torch.nn.utils.fuse_conv_bn_eval(net.conv3, net.bn3)
        net.bn3 = torch.nn.Identity()

def quantize(net):
    quant.convert(net, inplace=True)

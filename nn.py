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
        x = self.quant(x)
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        x = self.drop3(self.relu3(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        x = self.dequant(x)
        return x

THRESHOLD = 0.7

def validate(nn, threshold, data_loader):
    for n in nn: n.eval()
    # array of correct count for each net for each threshold
    correct_pos = np.zeros((len(nn), len(threshold)), dtype=np.int32)
    correct_neg = np.zeros((len(nn), len(threshold)), dtype=np.int32)
    with torch.no_grad():
        total = {"pos": 0, "neg": 0}
        for data in tqdm.tqdm(data_loader, desc="Validation", unit="Batches"):
            wavs, labels = data
            for idx, n in enumerate(nn):
                outputs = n(wavs)
                sig = torch.sigmoid(outputs[:,0])
                for i, t in enumerate(threshold):
                    predicted = sig > t
                    correct_pos[idx, i] += ((predicted == labels) and (labels == 1)).sum().item()
                    correct_neg[idx, i] += ((predicted == labels) and (labels == 0)).sum().item()
            total["pos"] += (labels == 1).sum().item()
            total["neg"] += (labels == 0).sum().item()
    for n in nn: n.train()
    tot_p = total["pos"]
    tot_n = total["neg"]
    for idx in range(len(nn)):
        print(f"Model #{idx}")
        for cp, cn, t in zip(correct_pos[idx], correct_neg,threshold):
            acc = (cp+cn) / (tot_p+tot_n)
            pos = cp / tot_p
            neg = cn / tot_n
            print(f"Threshold: {t} "
                  f"Acc {cp+cn}/{tot_p+tot_n} : {acc*100:.2f}% "
                  f"Pos: {cp}/{tot_p} : {pos*100:.2f} "
                  f"Neg: {cn}/{tot_n} : {neg*100:.2f}")

def train_loop(net, lr, w, loader):
    weights = torch.tensor([w], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
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
        loss = criterion(outputs, labels.float().unsqueeze(1))
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
    net.fc1 = torch.nn.utils.fuse_linear_bn_eval(net.fc1, net.bn1)
    net.bn1 = torch.nn.Identity()
    net.fc2 = torch.nn.utils.fuse_linear_bn_eval(net.fc2, net.bn2)
    net.bn2 = torch.nn.Identity()
    net.fc3 = torch.nn.utils.fuse_linear_bn_eval(net.fc3, net.bn3)
    net.bn3 = torch.nn.Identity()

def quantize(net):
    quant.convert(net, inplace=True)

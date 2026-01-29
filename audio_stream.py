import nn
import torch.ao.quantization as quant
from audio_preprocessor import *
import torch
from time import time, sleep
import pyaudio
import numpy as np
from test import MathNet

if __name__ == "__main__":
    net = nn.Net()
    # net.load_state_dict(torch.load("out/model_iter0.pth", weights_only=True))
    if True:
        quant.prepare_qat(net, inplace=True)
        net.load_state_dict(torch.load("out/model_iter0.pth", weights_only=True))
        nn.quantize(net)
        net.eval()
        math_net = MathNet(net)
    else:
        net.eval()
        net.load_state_dict(torch.load("model.pth", weights_only=True))
    mfcc_inst = MFCC()

    if True:
        audio = pyaudio.PyAudio()
        mic_stream = audio.open(rate=16000,
                                channels=1,
                                format=pyaudio.paInt16,
                                input=True,
                                frames_per_buffer=512)
        frames = np.zeros((nn.FRAMES_PER_SEC, nn.COEF_PER_FRAME), dtype=np.float32)
        print("Starting")
        print("Nothing     ", end='')
        last = False
        timestamp = 0
        pred_count = 0
        while True:
            aud = mic_stream.read(512)
            new_frame = np.frombuffer(aud, dtype=np.int16)
            frames[:-1] = frames[1:]
            # frames[-1] = mfcc_inst(new_frame)
            frames[-1] = mfcc(new_frame)
            features = torch.tensor(frames.flatten()).unsqueeze(0)
            with torch.no_grad():
                # out = net(features)
                out = math_net(features)
                pred = torch.sigmoid(out).item()
            # print(pred, pred > 0.9)
            if np.isnan(frames).any():
                break
            if pred > 0.7:
                pred_count += 1
            else:
                pred_count = 0
            if pred_count > 5:
                if last == False:
                    print("\rDETECTED           ", end='')
                    last = True
                    timestamp = time()
            elif time() - 1 > timestamp:
                print(f"\r{out.item():.3f} {pred*100:.2f}%     ", end='')
                last = False
    else:
        audio = load_wav("output.wav")
        pred = []
        with torch.no_grad():
            total_time = 0
            total = 0
            for idx in range(0, len(audio)-16000, 1000):
                total += 1
                prev = time()
                frames = prepare_data(audio[idx:])
                mfcc_data = process_frames(frames, mfcc_inst)
                # mfcc_data = process_frames(frames, mfcc_inst=None)
                # print(mfcc_data)
                features = torch.tensor(mfcc_data).unsqueeze(0)
                pred.append(torch.sigmoid(net(features)))
                total_time += time() - prev

        print(f"Avg time: {total_time / total}")
        pred = [val.item() for val in pred]
        print(pred)
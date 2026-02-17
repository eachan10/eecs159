import nn
import torch.ao.quantization as quant
from audio_preprocessor import *
import torch
from time import time, sleep
import pyaudio
import numpy as np
from test import MathNet

if __name__ == "__main__":
    net = nn.ConvNetStopGo()
    # net.load_state_dict(torch.load("out/model_iter0.pth", weights_only=True))
    if True:
        quant.prepare_qat(net, inplace=True)
        net.load_state_dict(torch.load("out-conv-stop-go/model_iter8.pth", weights_only=True))
        nn.fuse_module(net)
        nn.quantize(net)
        net.eval()
    else:
        net.eval()
        net.load_state_dict(torch.load("model.pth", weights_only=True))

    if True:
        audio = pyaudio.PyAudio()
        mic_stream = audio.open(rate=16000,
                                channels=1,
                                format=pyaudio.paInt16,
                                input=True,
                                frames_per_buffer=300)
        # frames = np.zeros((nn.FRAMES_PER_SEC, nn.COEF_PER_FRAME), dtype=np.float32)
        audio = np.zeros(16000)
        print("Starting")
        print("Nothing     ", end='')
        last = False
        timestamp = 0
        pred_count = 0
        while True:
            for i in range(10): # PARAM: number of frames to read before doing new inference
                aud = mic_stream.read(300)
                new_frame = np.frombuffer(aud, dtype=np.int16)
                audio[:-300] = audio[300:]
                audio[-300:] = new_frame
            # frames[-1] = mfcc_inst(new_frame)
            audio_frames = prepare_data(audio)
            features = np.expand_dims(process_frames(audio_frames), axis=0)  # adds channel
            features = torch.tensor(features).unsqueeze(0)  # adds batch
            with torch.no_grad():
                out = net(features)
                # out = math_net(features)
                prob = torch.softmax(out,dim=1)
                pred = torch.argmax(prob, dim=1).item()
            threshold = [2, 0, 0.8]
            if prob[0][pred] > threshold[pred]:    # PARAM: threshold for sigmoid function - decrease makes more likely to be positive
                pred_count += 1
            else:
                pred_count = 0
            if pred_count >= 1: # PARAM: number of consecutive positives needed to detect for actual positive
                if last == 0:
                    print(f"\rDETECTED {pred}                              ", end='')
                    last = pred
                    timestamp = time()
            elif time() - 1 > timestamp:
                print(f"\r{prob[0][0].item():.2f} {prob[0][1].item():.2f} {prob[0][2].item():.2f} {pred}     ", end='')
                last = 0

import pyaudio
import numpy as np

import wave

import nn
import torch.ao.quantization as quant
from audio_preprocessor import *
import torch
from time import time, sleep
import pyaudio
import numpy as np
import timeit

if __name__ == "__main__":
    # time the different models
    torch.backends.quantized.engine = 'qnnpack'
    net = nn.ConvNetStopGo()
    quant.prepare_qat(net, inplace=True)
    net.load_state_dict(torch.load("out-conv-stop-go/model_iter8.pth", weights_only=True))
    nn.fuse_module(net)
    nn.quantize(net)
    net.eval()
    inp = torch.rand(1000, 1, 52, 12)
    start = time()
    with torch.no_grad():
        for i in range(1000):
            out = net(inp[i:i+1])
    stop = time()
    print(f"ConvNetStopGo: {stop - start:.4f} seconds")
    print(f"ConvNetStopGo per inference: {(stop - start)/1000:.6f} seconds")

    net = nn.ConvNetStopGo()
    net.load_state_dict(torch.load("out-conv-stop-go-float/model_iter8.pth", weights_only=True))
    net.eval()
    start = time()
    with torch.no_grad():
        for i in range(1000):
            out = net(inp[i:i+1])
    stop = time()
    print(f"ConvNetStopGo-float: {stop - start:.4f} seconds")
    print(f"ConvNetStopGo-float per inference: {(stop - start)/1000:.6f} seconds")

    start = time()
    inp = np.random.random((1000, 512))
    for i in range(1000):
        out = mfcc_fast(inp[i])
    stop = time()
    print(f"MFCC: {stop - start:.4f} seconds")
    print(f"MFCC per inference: {(stop - start)/1000:.6f} seconds")
    # p = pyaudio.PyAudio()
    # mic_stream = p.open(rate=48000,
    #                     channels=1,
    #                     format=pyaudio.paInt16,
    #                     input=True,
    #                     frames_per_buffer=300,
    #                     input_device_index=0)
    
    # aud = mic_stream.read(48000*10)
    # mic_stream.stop_stream()
    # mic_stream.close()
    # p.terminate()
    # wf = wave.open("out.wav", 'wb')
    # wf.setnchannels(1)
    # wf.setsampwidth(2)
    # wf.setframerate(48000)
    # wf.writeframes(aud)
    # wf.close()
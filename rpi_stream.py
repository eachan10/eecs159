import nn
import torch.ao.quantization as quant
from audio_preprocessor import *
import torch
from time import time, sleep
import pyaudio
import numpy as np

import gpiozero

EN_GPIO = False
if __name__ == "__main__":
    if EN_GPIO:
        motor_en = gpiozero.LED(12)
        motor_a1 = gpiozero.LED(1)

    # Load trained model and quantize for inference
    torch.backends.quantized.engine = 'qnnpack'
    net = nn.ConvNetStopGo()
    quant.prepare_qat(net, inplace=True)
    # net.load_state_dict(torch.load("out-cnn-32-64-128-fc1/model_iter9.pth", weights_only=True))
    net.load_state_dict(torch.load("out-conv-stop-go/model_iter9.pth", weights_only=True))
    nn.fuse_module(net)
    nn.quantize(net)
    net.eval()

    # open audio stream from i2s microphone
    audio = pyaudio.PyAudio()
    for i in range(audio.get_device_count()):
        dev = audio.get_device_info_by_index(i)
        if "googlevoicehat" in dev["name"]:
            dev_index = i
            break
    
    # number of frames to read before doing inference
    FRAMES_PER_INFERENCE = 10
    mic_stream = audio.open(rate=48000,
                            channels=1,
                            format=pyaudio.paInt16,
                            input=True,
                            frames_per_buffer=900 * FRAMES_PER_INFERENCE,
                            input_device_index=dev_index)

    # used to track when last positive detection was to prevent spamming detections
    timestamp = 0
    # whether last prediction was positive or not to prevent spamming detections
    pred_count = 0
    # features buffer to hold MFCC features for the last 1 second of audio (52 frames)
    features = torch.zeros((1, 1, 52, 12)).to(torch.float32)
    # audio buffer to hold most recent audio for MFCC calculation
    audio = np.zeros(300 * FRAMES_PER_INFERENCE + 100, dtype=np.int16)
    # buffer for calculating MFCC for one frame
    frame_buf = np.zeros(512, dtype=np.float64)

    print("Starting")
    print("Nothing     ", end='')

    if EN_GPIO:
        motor_a1.off()
        motor_en.on()
    try:
        with torch.no_grad():
            while True:
                # read new audio frames from microphone
                aud = mic_stream.read(900 * FRAMES_PER_INFERENCE)
                new_frame = np.frombuffer(aud, dtype=np.int16)

                # shift old audio to left and add new audio to end of buffer
                audio[:-300 * FRAMES_PER_INFERENCE] = audio[300 * FRAMES_PER_INFERENCE:]
                audio[-300 * FRAMES_PER_INFERENCE:] = resample_poly(new_frame, 16000, 48000)

                # calculate only new MFCC features for the new frames and concatenate with old features
                # shift old features to left
                for i in range(FRAMES_PER_INFERENCE):
                    # calculate MFCC for new frame and add to features
                    frame_buf[:400] = audio[i*300:(i*300)+400] * hamming
                    out = mfcc_fast(frame_buf)
                    out = (out - out.mean()) / (out.std() + 1e-8)
                    features[0, 0, i, :] = torch.from_numpy(out)
                features = features.roll(-FRAMES_PER_INFERENCE, dims=2) # shift features to left by number of new frames

                # only do inference if it's been at least 1 second since last positive detection
                if time() - 1 > timestamp:

                    # extract prediction and confidence from model output
                    out = net(features)
                    prob = torch.softmax(out,dim=1)[0]
                    pred = torch.argmax(prob, dim=0).item()
                    threshold = [2, 0.4, 0.5]
                    # if prob[pred] > threshold[pred]:
                    #     pred_count += 1
                    # else:
                    #     pred_count = 0
                    # if pred_count >= 1: # PARAM: number of consecutive positives needed to detect for actual positive
                    if prob[1] > threshold[1]:
                        # stop detected
                        print(f"\rDETECTED STOP                              ", end='')
                        if EN_GPIO:
                            motor_a1.off()
                        timestamp = time()
                    elif prob[2] > threshold[2]:
                        # go detected
                        print(f"\rDETECTED GO                              ", end='')
                        if EN_GPIO:
                            motor_a1.on()
                        timestamp = time()
                    else:
                        continue
                        print(f"\r{prob[0].item():.2f} {prob[1].item():.2f} {prob[2].item():.2f} {pred}     ", end='')
    finally:
        if EN_GPIO:
            motor_a1.off()
            motor_en.off()

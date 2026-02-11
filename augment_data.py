"""Add the augmentations to the data samples"""
from pathlib import Path
from pydub import AudioSegment
import random
import math
import time

random.seed(time.time())

def augment(audio: AudioSegment, noise):
    # random gain +-12dB
    audio = audio.apply_gain(random.randint(-12, 12))

    # noise
    # overlay noise from random segment of random noise choice
    # random psnr 0-20 dB
    snr = random.randint(0, 20)

    # get 1000ms slice of noise
    noise_sample = random.choice(noise)
    duration = len(noise_sample)  # length in ms
    start = random.randint(0, duration-1000)
    noise_slice = noise_sample[start:start+1000] # 1000ms slice

    # calculate noise gain with target snr
    signal_rms = audio.rms
    noise_rms = noise_slice.rms
    if noise_rms == 0:  # noise is silence
        noise_gain = 0
    elif signal_rms == 0:  # signal is silence
        noise_gain = random.randint(-2, 5)
    else:
        noise_rms_target = signal_rms / (10 ** (snr / 20))
        noise_gain = 20 * math.log10(noise_rms_target / noise_rms)

    # timeshift +- 100ms
    # reverb?
    # speed/pitch
    timeshift = random.randint(-100, 100)

    if timeshift >= 0:
        return noise_slice.overlay(audio, gain_during_overlay=noise_gain, position=timeshift)
    else:
        return noise_slice.overlay(audio[-timeshift:], gain_during_overlay=noise_gain)

def zero_p(audio: AudioSegment):
    frame_len = 25
    rms = [audio[idx:idx+400].rms for idx in range(0, len(audio), frame_len)]
    peak = max(rms)
    thres = peak // 5
    word_middle = rms.index(peak)
    # find index after middle where word ends
    for i in range(word_middle, len(rms)):
        if rms[i] <= thres:
            break
    i -= 3
    if i < 0: i = 0
    clip = audio[:i*frame_len].fade_out(50) #+ AudioSegment.silent(duration=1000-i*frame_len, frame_rate=16000)
    clip.export("output.wav", format="wav")
    return rms

def main2():
    import random
    with open("val_pos_set.txt") as f:
        fps = [l.strip() for l in f.readlines() if len(l)>5]
    aud = AudioSegment.from_file(f"speech-data/{random.choice(fps)}") 
    print(zero_p(aud))

def main():
    noise = []
    for p in Path("./speech-data/_background_noise_").iterdir():
        if not p.name.endswith(".wav"): continue
        noise.append(AudioSegment.from_file(p))
    dirs = [p for p in Path("./speech-data").iterdir() if p.is_dir()]
    # for dir_ in dirs:
        # if dir_.name == "_background_noise_": continue
        # if "augmented" in dir_.name: continue
        # for p in dir_.iterdir():
            # if not p.name.endswith(".wav"): continue
            # count = 10 if dir_.name == "stop" else 3
            # for n in range(count):
            #     sample = AudioSegment.from_file(p)
            #     sample = augment(sample, noise)
                
                # if dir_.name == "stop":
                #     # save to stop_augmented
                #     fp = f"speech-data/stop-augmented/{p.stem}_{dir_.name}_aug{n}.wav"
                #     sample.export(fp, format='wav')
                # else:
                #     # save to negative_augmented
                #     fp = f"speech-data/neg-augmented/{p.stem}_{dir_.name}_aug{n}.wav"
                #     sample.export(fp, format='wav')
    for idx, n in enumerate(noise):
        for samp in range(15000):
            duration = len(n)
            start = random.randint(0, duration-1000)
            sample = n[start:start+1000]
            sample += random.randint(-12, 12)
            fp = f"speech-data/neg-augmented/noise{idx}-{samp}.wav"
            sample.export(fp, format='wav')



if __name__ == "__main__":
    main()

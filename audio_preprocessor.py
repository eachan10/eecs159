import numpy as np
from scipy.fft import dct
from scipy.signal import resample_poly
import matplotlib.pyplot as plt

import cmsisdsp
import cmsisdsp.mfcc
from cmsisdsp.datatype import F32

import wave


def hz2mel(hz):
    return 2595 * np.log10(1+hz/700.0)
def mel2hz(mel):
    return 700*(10**(mel/2595.0)-1)

# low and high frequencies for mel filterbanks
LOW_FREQ = 0
HIGH_FREQ = 8000
BINS = 36
endpoints = np.linspace(hz2mel(LOW_FREQ), hz2mel(HIGH_FREQ), BINS+2)
endpoints = mel2hz(endpoints) # endpoints in hz
endpoints = np.floor((512+1)/16000*endpoints) # endpoints in fft indices
# each filterbank will be a 257 point array to multiply the fft by
# the nonzero portion of the filterbank is a triangular function
filterbanks = np.zeros((BINS, 257))

for j in range(BINS):
    start = int(endpoints[j])
    middle = int(endpoints[j+1])
    end = int(endpoints[j+2])
    for i in range(start, middle):
        filterbanks[j,i] = (i-start) / (middle-start)
    for i in range(middle, end):
        filterbanks[j,i] = (end-i) / (end-middle)

# audio recorded as 16kHz
# pre-emphasis
def pre_emphasis(signal: np.ndarray):
    COEF = 0.95   # usual range from 0.95 to 0.97
    return np.append(signal[0], signal[1:] - COEF * signal[:-1])

def power_spectrum(signal: np.ndarray):
    # takes a 400 sample frame
    # 512 point fft of it
    # compute the magnitude spectrum by taking the 512-point fft
    # taking real fft
    mag_spec = np.abs(np.fft.rfft(signal, 512))
    # compute the power spectrum
    pow_spec = 1.0 / 512 * np.square(mag_spec)
    return pow_spec

def filterbank(signal):
    s = pre_emphasis(signal)
    pow_spec = power_spectrum(s)
    energies = np.dot(pow_spec, filterbanks.T)

    # replace zeros with minimum value float so log doesn't fail
    energies = np.where(energies == 0, np.finfo(float).eps, energies)
    energies = np.log(energies)
    return energies

def mfcc(signal):
    energies = filterbank(signal)
    features = dct(energies, type=2, axis=0, norm='ortho')
    # return features # only need the lower 12 for audio speech recognition

    # apply cepstral lifter
    L = 22.0
    n = np.arange(12)
    lift = 1 + L/2 * np.sin(np.pi*n/L)
    features:np.ndarray = features[:12] * lift
    return features # only need the lower 12 for audio speech recognition

hamming = 0.54 - 0.46 * np.cos(2*np.pi*np.arange(0,400)/(400-1))
# the number of frames is the len of this range
# 16000 samples per second for a 1 second clip
# 400 is the length of one frame
# 300 is the step size between frames
frames = len(range(0, 16000 - 400, 300))
def load_wav(path, downsample=False):
    with wave.open(path) as wf:
        n_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        audio_bytes = wf.readframes(n_frames)
        framerate = wf.getframerate()
    
    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        raise RuntimeError()
    audio = np.frombuffer(audio_bytes, dtype=dtype)

    if downsample:
        audio = resample_poly(audio, 16000, framerate)
    return audio
 
def prepare_data(audio: np.ndarray):
    """prepare a 1s or about 1s audio wav file into flattened features"""
    if len(audio) >= 16000:
        audio = audio[:16000]
    else:
        audio = np.pad(audio, 16000-len(audio), mode='constant', constant_values=0)
    audio_frames = np.zeros((frames,512), dtype=np.float32)
    for i in range(frames):
        audio_frames[i,0:400] = audio[i*300:i*300+400] * hamming
    return audio_frames

def process_frames(audio_frames, mfcc_inst=None):
    out = np.zeros((frames, 12), dtype=np.float32)
    for idx, frame in enumerate(audio_frames):
        if mfcc_inst:
            out[idx] = mfcc_inst(frame)
        else:
            out[idx] = mfcc(frame)
    # normalize: cepstral mean variance normalization
    out = (out - out.mean()) / (out.std() + 1e-8)
    return out

SAMPLE_RATE = 16000
FFT_SIZE = 512
N_DCT_OUTPUTS = 12
FREQ_MIN = 0
FREQ_MAX = SAMPLE_RATE // 2
N_MEL_FILTERS = 36

class MFCC:
    def __init__(self):
        self.sample_rate = 16000
        self.fft_size = 512
        self.n_dct_outputs = 12
        self.freq_min = 40
        self.freq_high = self.sample_rate // 2
        self.n_mel_filters = 26
        self.window = hamming
        
    def __call__(self, frame):
        '''Frame must be a 512 length np array of f32'''
        self.mfccf32 = cmsisdsp.arm_mfcc_instance_f32()
        self.filtLen, self.filtPos, self.packedFilters = cmsisdsp.mfcc.melFilterMatrix(F32,
                                                               self.freq_min,
                                                               self.freq_high,
                                                               self.n_mel_filters,
                                                               self.sample_rate,
                                                               self.fft_size)
        self.dctMatrixFilters = cmsisdsp.mfcc.dctMatrix(F32,
                                          self.n_dct_outputs,
                                          self.n_mel_filters)
        status = cmsisdsp.arm_mfcc_init_f32(self.mfccf32,
                                            self.fft_size,
                                            self.n_mel_filters,
                                            self.n_dct_outputs,
                                            self.dctMatrixFilters,
                                            self.filtPos, self.filtLen, self.packedFilters, self.window)
        assert status == 0
        frame = pre_emphasis(frame)
        tmp_nb = cmsisdsp.arm_mfcc_tmp_buffer_size(F32, self.fft_size,1)
        tmp = np.zeros(tmp_nb, dtype=np.float32)
        res = cmsisdsp.arm_mfcc_f32(self.mfccf32, frame, tmp)
        return res
# 16kHz 25ms frame is 400 samples

if __name__ == "__main__":
    audio = load_wav('../output.wav')
    # audio = load_wav("speech-data/go/0a2b400e_nohash_0.wav")
    print(max(audio))
    print(f'{len(audio)=}')
    out = []
    out2 = []
    my_mfcc = MFCC()
    for i in range(0, len(audio), 16000):
        audio_frames = prepare_data(audio[i:i+16000])
        for f in audio_frames:
            out.append(my_mfcc(f))
            out2.append(mfcc(f))
    print(out[0])
    print(out2[0])
    print("MFCC done")
    fig, ax = plt.subplots(1, 2)
    for_plot = np.array(out2)
    print(for_plot.shape)
    im1 = ax[0].imshow(for_plot.T, aspect='auto')
    ax[0].set_ylabel("MFCC Coef")
    ax[0].set_xlabel("Frame Index")

    import python_speech_features
    out = python_speech_features.mfcc(audio.T, appendEnergy=False, ceplifter=22)
    im2 = ax[1].imshow(out.T, aspect='auto')
    # audio, sr = torchaudio.load("speech-data/go/0a2b400e_nohash_0.wav")
    # transform = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=12,
                                        #    melkwargs={"n_fft": 512, "hop_length": 400,
                                                    #   "n_mels": 26})
    # out = transform(audio)
    # im2 = ax[1].imshow(out[0], aspect='auto')
    ax[1].set_ylabel("MFCC Coef")
    ax[1].set_xlabel("Frame Index")

    fig.colorbar(im1, ax=ax[0])
    fig.colorbar(im2, ax=ax[1])
    plt.tight_layout()
    plt.show()
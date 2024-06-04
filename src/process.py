import torch
import torchaudio
import numpy as np
from PIL import Image
import scipy.signal as signal
import pyreaper


device = 'cuda' if torch.cuda.is_available() else 'cpu'


SAMPLE_RATE = 44100
N_FFT = 17640
HOP_LENGTH = 441
WIN_LENGTH = 4410
F_MIN = 0
F_MAX = 10000
N_MELS = 512

spectrogram_func = torchaudio.transforms.Spectrogram(
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    pad=0,
    window_fn=torch.hann_window,
    power=None,
    normalized=False,
    wkwargs=None,
    center=True,
    pad_mode="reflect",
    onesided=True,
).to(device)


inverse_spectrogram_func = torchaudio.transforms.GriffinLim(
    n_fft=N_FFT,
    n_iter=64, #32
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    window_fn=torch.hann_window,
    power=1.0,
    wkwargs=None,
    momentum=0.9,
    length=None,
    rand_init=True,
).to(device)


mel_scaler = torchaudio.transforms.MelScale(
    n_mels=N_MELS,
    sample_rate=SAMPLE_RATE,
    f_min=F_MIN,
    f_max=F_MAX,
    n_stft = N_FFT//2 + 1,
    norm = None,
    mel_scale = "htk"
).to(device)

# inverse_mel_scaler = torchaudio.transforms.InverseMelScale(
#     n_stft=N_FFT // 2 + 1,
#     n_mels=N_MELS,
#     sample_rate=SAMPLE_RATE,
#     f_min=F_MIN,
#     f_max=F_MAX,
#     norm=None,
#     mel_scale="htk",
# ).to(device)


inverse_mel_scaler = torchaudio.transforms.InverseMelScale(
    n_stft=N_FFT // 2 + 1,
    n_mels=N_MELS,
    sample_rate=SAMPLE_RATE,
    f_min=F_MIN,
    f_max=F_MAX,
    max_iter=200,
    tolerance_loss=1e-5,
    tolerance_change=1e-8,
    sgdargs=None,
    norm=None,
    mel_scale="htk",
).to(device)
"""
Encoding Pipelines
"""
 
def waveform_to_vae_input(waveform):
    """Full Encoding Pipeline"""
    image, max_value = image_from_spectrogram(spectrogram_from_waveform(waveform))
    return preprocess_image(image), max_value

def waveform_to_image(waveform):
    return image_from_spectrogram(spectrogram_from_waveform(waveform))


def spectrogram_from_waveform(waveform):
    waveform_tensor = torch.from_numpy(waveform.astype(np.float32)).to(device)
    spectrogram_complex = spectrogram_func(waveform_tensor)
    amplitudes = torch.abs(spectrogram_complex)
    return mel_scaler(amplitudes).cpu().numpy()

def image_from_spectrogram(spectrogram, power = 0.25):
    max_value = np.max(spectrogram)
    data = spectrogram / max_value
    data = np.power(data, power)
    data = data * 255
    data = 255 - data
    data = data.astype(np.uint8)
    image = Image.fromarray(data, mode="L").convert("RGB")
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return image, max_value

def preprocess_image(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))
    image = image.resize((w, h), resample=Image.LANCZOS)
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = image_np[None].transpose(0, 3, 1, 2)
    image_torch = torch.from_numpy(image_np)
    return 2.0 * image_torch - 1.0


"""
Decoding Pipeline
"""

def vae_output_to_waveform(decoder_output, pipe, max_value):
    """Full Decoding Pipeline"""
    return waveform_from_spectrogram(spectrogram_from_image(decoder_output_to_image(decoder_output, pipe), max_value=max_value))

def image_to_waveform(image, max_value):
    return waveform_from_spectrogram(spectrogram_from_image(image, max_value=max_value))

def decoder_output_to_image(decoder_output, pipe):
    image = (decoder_output.detach() / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    return pipe.numpy_to_pil(image)[0]

def spectrogram_from_image(image, max_value=30e6, power=0.25):
    if image.mode in ("P", "L"):
        image = image.convert("RGB")
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    data = np.array(image).transpose(2, 0, 1)
    data = data[0:1, :, :]
    data = data.astype(np.float32)
    data = 255 - data
    data = data / 255
    data = np.power(data, 1 / power)
    data = data * max_value
    return data[0]


# Low pass filter
fir_coeff = signal.firwin(255, 6000, fs=SAMPLE_RATE)

def waveform_from_spectrogram(mel_amplitudes, filter=True):
    mel_amplitudes_torch = torch.from_numpy(mel_amplitudes).to(device)
    amplitudes_linear = inverse_mel_scaler(mel_amplitudes_torch)
    waveform = inverse_spectrogram_func(amplitudes_linear)
    waveform = waveform.detach().cpu().numpy()
    waveform = signal.filtfilt(fir_coeff, 1.0, waveform)
    return waveform



"""
PROCESSING LENGTH
"""
def truncate_real_sound(x, length=253575):
    if x.shape[-1] == length:
        return x
    elif x.shape[-1] < length:
        result = np.zeros((length))
        result[x.shape[-1]:] = x
    elif x.shape[-1] > length:
        result = x[:length]
    return result

"""
GATING
"""
TOTAL_LENGTH = 6
N_TIME_FRAMES = 576

freq_banks = torchaudio.functional.melscale_fbanks(n_freqs=SAMPLE_RATE//2,
    f_min=F_MIN, f_max=F_MAX, n_mels=N_MELS, sample_rate=SAMPLE_RATE).numpy()

hann_window = torch.hann_window(WIN_LENGTH).numpy()
time_banks = np.zeros((SAMPLE_RATE*TOTAL_LENGTH, N_TIME_FRAMES)) # 6 seconds, 576 time-frames


for i in range(N_TIME_FRAMES):
    time_banks[i*HOP_LENGTH:i*HOP_LENGTH+WIN_LENGTH, i] = hann_window

def get_freq_bin_activations(freqs):
    return freq_banks[np.round(freqs).astype(int)]

def get_time_bin_activations(times):
    return time_banks[np.round(times).astype(int)]

def gen_spectrogram_from_activations(times, freqs):
    time_bin_activations = np.expand_dims(get_time_bin_activations(times*44100), -2)
    freq_bin_activations = np.expand_dims(get_freq_bin_activations(freqs), -1)
    return np.sum(time_bin_activations * freq_bin_activations, axis=0)


def gen_harmonic_spectrogram(x, n_freqs=512, n_time_steps=576):
    x = truncate_real_sound(x)
    _, _, f0_times, f0_freqs, _ = pyreaper.reaper(np.int16(65536*x), fs=SAMPLE_RATE)

    highest_frequency = np.max(f0_freqs)
    highest_harmonic = int(F_MAX//highest_frequency)
    harmonics = np.array([f0_freqs*i for i in range(1,highest_harmonic)])

    spec = np.zeros((512,576))
    count = 0
    for harmonic in harmonics:
        spec += gen_spectrogram_from_activations(f0_times, harmonic)
        #print(count)
        count += 1
    return (spec==0).astype(np.float32)

def gen_gated_spectrogram(x, threshold=0.001):
    x = truncate_real_sound(x)
    spectrogram = spectrogram_from_waveform(x)
    return (spectrogram<np.max(spectrogram)*threshold).astype(np.float32)



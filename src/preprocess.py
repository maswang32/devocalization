import torch
import torchaudio
import numpy as np
from PIL import Image
import scipy.signal as signal

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
fir_coeff = signal.firwin(255, 10000, fs=SAMPLE_RATE)

def waveform_from_spectrogram(mel_amplitudes, filter=True):
    mel_amplitudes_torch = torch.from_numpy(mel_amplitudes).to(device)
    amplitudes_linear = inverse_mel_scaler(mel_amplitudes_torch)
    waveform = inverse_spectrogram_func(amplitudes_linear)
    waveform = waveform.detach().cpu().numpy()
    waveform = signal.filtfilt(fir_coeff, 1.0, waveform)
    return waveform

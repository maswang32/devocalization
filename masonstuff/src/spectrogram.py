import torch
import torchaudio 
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MelConverter:
    def __init__(self, hop_length=86, n_mels=512, n_fft=8192, win_length=2048, fs=22050):
        
        self.spec_func = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length, #256
            win_length=win_length,
            pad=0,
            window_fn=torch.hann_window,
            power=None,
            normalized=False,
            wkwargs=None,
            center=True,
            pad_mode="reflect",
            onesided=True,
        ).to(device)

        self.mel_scaler = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=fs,
            f_min=0,
            f_max=10000,
            n_stft=n_fft//2 + 1,
            norm=None,
            mel_scale="htk",
        ).to(device)

        self.inv_spec_func = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            n_iter=12,
            win_length=win_length,
            hop_length=hop_length, # 256
            window_fn=torch.hann_window,
            power=1.0,
            wkwargs=None,
            momentum=0.99,
            length=None,
            rand_init=True,
        ).to(device)

        self.inv_mel_scaler = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft//2 + 1,
            n_mels=n_mels,
            sample_rate=fs,
            f_min=0,
            f_max=10000,
            max_iter=200,
            tolerance_loss=1e-5,
            tolerance_change=1e-8,
            sgdargs=None,
            norm=None,
            mel_scale="htk",
        ).to(device)

    def audio_to_mel(self, x):
        return self.mel_scaler(torch.abs(self.spec_func(x)))

    def mel_to_audio(self, mel_amplitudes):
        return self.inv_spec_func(self.inv_mel_scaler(mel_amplitudes))


# Example Converters
conv_2s_to_512_512 = MelConverter(hop_length=86, n_mels=512, n_fft=8192, win_length=2048, fs=22050)
conv_2s_to_256_256 = MelConverter(hop_length=172, n_mels=256, n_fft=8192, win_length=2048, fs=22050)



def save_mel_spectrograms(wav_load_path, spec_save_path, converter=conv_2s_to_256_256, image_size = 256):
    waveforms = torch.from_numpy(np.load(wav_load_path)).to(device)
    n_data = waveforms.shape[0]
    spectrograms = torch.zeros((n_data, image_size, image_size)).to(device)
    
    for i in range(n_data):
        spectrograms[i] = converter.audio_to_mel(waveforms[i])[:image_size, :image_size]
        if i%100 == 0:
            print(i)
    
    spectrograms = spectrograms.cpu().numpy()
    np.save(spec_save_path, spectrograms)




    
import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_mel_spec(y, fs=44100, n_fft=2048, hop_length=256, n_mels=128, db=True, title='Mel-frequency spectrogram'):
    fig, ax = plt.subplots()
    S = librosa.feature.melspectrogram(y=y, sr=fs, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length)
    if db:
        S_to_plot = librosa.power_to_db(S)
    else:
        S_to_plot = S
    img = librosa.display.specshow(S_to_plot, x_axis='time',
                         y_axis='mel', sr=fs,
                         fmax=fs//2, ax=ax, hop_length=hop_length)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=title)
    return S_to_plot
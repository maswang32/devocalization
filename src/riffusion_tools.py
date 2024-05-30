import sys
sys.path.insert(1, '/viscam/projects/audio_nerf/transfer/audio_nerf/riffusion-inference')

from diffusers import DiffusionPipeline
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
from io import BytesIO
from IPython.display import Audio
import torch
import torchaudio
import numpy as np
from scipy.io import wavfile
import numpy as np
from PIL import Image

pipe = DiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1")
pipe = pipe.to("cuda")



import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
# from denoiser import Denoiser

import librosa
from waveglow.glow import WaveGlow

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')

def save_wav(x, path) :
    librosa.output.write_wav(path, x.astype(np.float32), sr=hparams.sampling_rate)

# Setup hparams
hparams = create_hparams()
hparams.sampling_rate = 22050

# Load model from checkpoint
checkpoint_path = "tacotron2_statedict.pt"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

# Load WaveGlow for mel2audio synthesis and denoiser
waveglow_path = 'waveglow_256channels.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
# denoiser = Denoiser(waveglow)

# Prepare text input
text = "I don't know how to speak chinese."
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()

# Decode text input and plot results
mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

# Synthesize audio from spectrogram using WaveGlow
with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
print('save voice ...')
save_wav(audio[0].data.cpu().numpy(), 'test.wav')

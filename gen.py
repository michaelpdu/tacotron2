import sys
sys.path.append('waveglow/')
import numpy as np
import torch
import os
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
import librosa
from waveglow.glow import WaveGlow
import argparse

class TTSHelper:
    """"""

    def __init__(self):
        # Setup hparams
        self.hparams = create_hparams()
        self.hparams.sampling_rate = 22050

        # Load model from checkpoint
        checkpoint_path = "tacotron2_statedict.pt"
        self.model = load_model(self.hparams)
        self.model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        _ = self.model.cuda().eval().half()

        # Load WaveGlow for mel2audio synthesis and denoiser
        waveglow_path = 'waveglow_256channels.pt'
        self.waveglow = torch.load(waveglow_path)['model']
        self.waveglow.cuda().eval().half()
        for k in self.waveglow.convinv:
            k.float()

    def save_wav(self, x, path) :
        librosa.output.write_wav(path, x.astype(np.float32), sr=self.hparams.sampling_rate)

    def generate(self, text, voice_path):
        sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        # Decode text input and plot results
        _, mel_outputs_postnet, _, _ = self.model.inference(sequence)
        # Synthesize audio from spectrogram using WaveGlow
        with torch.no_grad():
            audio = self.waveglow.infer(mel_outputs_postnet, sigma=0.666)
        print('save voice ...')
        self.save_wav(audio[0].data.cpu().numpy(), voice_path)

    def generate_by_sentences(self, text_path, voice_dir_path=None):
        if voice_dir_path is None:
            voice_dir_path = 'voice_gen_dir'
        if not os.path.exists(voice_dir_path):
            os.makedirs(voice_dir_path)
        index = 1   
        with open(text_path, 'r') as fh:
            for line in fh.readlines():
                self.generate(line, os.path.join(voice_dir_path, 'voice_{}.wav'.format(index)))
                index += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Usages of TTS Helper')
    parser.add_argument("-t", "--text", type=str, help="input text for voice translation")
    parser.add_argument("-i", "--input_file", type=str, help="input file for voice translation")
    parser.add_argument("-o", "--output", type=str, help="voice output")
    args = parser.parse_args()

    helper = TTSHelper()
    if args.text:
        helper.generate(args.text, args.output)
    elif args.input_file:
        helper.generate_by_sentences(args.input_file, args.output)
    else:
        parser.print_usage()
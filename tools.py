import torch
import torchaudio
from torch import Tensor
import numpy as np

def revise(sentence):
    words = sentence[0].split()
    result = []
    for word in words:
        tmp = ''    
        for t in word:
            if not tmp:
                tmp += t
            elif tmp[-1]!= t:
                tmp += t
        if tmp == '스로':
            tmp = '스스로'
        result.append(tmp)
    return ' '.join(result)
    

def parser(signal, audio_extension: str = 'pcm') -> Tensor:

    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)

    return torch.FloatTensor(feature).transpose(0, 1)

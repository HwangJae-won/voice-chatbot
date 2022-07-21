import os, argparse, glob, torch, scipy, librosa
import numpy as np
from jamo import hangul_to_jamo
import soundfile as sf
import IPython
from TTS_tacotron1.models.tacotron import Tacotron, post_CBHG
from TTS_tacotron1.util.text import text_to_sequence, sequence_to_text
from TTS_tacotron1.util.plot_alignment import plot_alignment
from TTS_tacotron1.models.modules import griffin_lim
from TTS_tacotron1.util.hparams import *

save_dir = './output'

def makeTTS1(model, text, idx):
    seq = text_to_sequence(text)
    enc_input = torch.tensor(seq, dtype=torch.int64).unsqueeze(0)
    sequence_length = torch.tensor([len(seq)], dtype=torch.int32)
    dec_input = torch.from_numpy(np.zeros((1, mel_dim), dtype=np.float32))
    
    pred, alignment = model(enc_input, sequence_length, dec_input, is_training=False, mode='inference')
    pred = pred.squeeze().detach().numpy()
    alignment = np.squeeze(alignment.detach().numpy(), axis=0)

    np.save(os.path.join(save_dir, "bot_answer"), pred, allow_pickle=False)

    input_seq = sequence_to_text(seq)
#    alignment_dir = os.path.join(save_dir, 'align-{}.png'.format(idx))
#    plot_alignment(alignment, alignment_dir, input_seq)

def makeTTS2(model, text, idx):
    mel = torch.from_numpy(text).unsqueeze(0)
    pred = model(mel)
    pred = pred.squeeze().detach().numpy() 
    pred = np.transpose(pred)
    
    pred = (np.clip(pred, 0, 1) * max_db) - max_db + ref_db
    pred = np.power(10.0, pred * 0.05)
    wav = griffin_lim(pred ** 1.5)
    wav = scipy.signal.lfilter([1], [1, -preemphasis], wav)
    wav = librosa.effects.trim(wav, frame_length=win_length, hop_length=hop_length)[0]
    wav = wav.astype(np.float32)
    sf.write(os.path.join(save_dir, 'bot_answer.wav'), wav, sample_rate)
    
def inference_tacotron1(text):
    TTS_CP = './TTS_tacotron1/ckpt/origincode/1/ckpt-400000.pt'  # pretrained CP location
    os.makedirs(save_dir, exist_ok=True)
    
    model_Tacotron = Tacotron(K=16, conv_dim=[128, 128])
    ckpt = torch.load(TTS_CP)
    model_Tacotron.load_state_dict(ckpt['model'])
    
    for i, text in enumerate(text):
        jamo = ''.join(list(hangul_to_jamo(text)))
        makeTTS1(model_Tacotron, jamo, i)

    #make audio and play
    Audio_CP = './TTS_tacotron1/ckpt/origincode/2/ckpt-50000.pt'  # pretrained CP location
    mel_list = glob.glob(os.path.join(save_dir, 'bot_answer.npy'))
    
    model_Audio = post_CBHG(K=8, conv_dim=[256, mel_dim])
    ckpt2 = torch.load(Audio_CP)
    model_Audio.load_state_dict(ckpt2['model'])
    
    for i, fn in enumerate(mel_list):
        mel = np.load(fn)
        makeTTS2(model_Audio, mel, i)
    IPython.display.display(IPython.display.Audio(os.path.join(save_dir, 'bot_answer.wav'), autoplay=True))
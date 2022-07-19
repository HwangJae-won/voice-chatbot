# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch import Tensor
import torchaudio
import numpy as np
import librosa
import os
from matplotlib import pyplot as plt
import librosa.display

from GetSpeech import get_speech
from tools import revise, parser
from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.models import DeepSpeech2
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

import argparse, glob, torch, scipy
import numpy as np
from jamo import hangul_to_jamo
import soundfile as sf
import IPython
from TTS_tacotron1.models.tacotron import Tacotron, post_CBHG
from TTS_tacotron1.util.text import text_to_sequence, sequence_to_text
from TTS_tacotron1.util.plot_alignment import plot_alignment
from TTS_tacotron1.models.modules import griffin_lim
from TTS_tacotron1.util.hparams import *

import warnings
warnings.filterwarnings('ignore')
from TTS_tacotron1.test_tacotron import inference_tacotron1

#device = 'cuda' if torch.cuda.is_available() else 'cpu'

device = 'cpu'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tokenizer = PreTrainedTokenizerFast.from_pretrained('byeongal/Ko-DialoGPT')
STT_vocab = KsponSpeechVocabulary('./aihub_character_vocabs.csv')

#Model list
STT_model_path = "./model_ds2.pt"

CH_model = GPT2LMHeadModel.from_pretrained('byeongal/Ko-DialoGPT').to(device)


past_user_inputs = []
generated_responses = []

n = 0
while n < 6:

    n += 1
    if n == 1:
        start_input = '안녕하세요 가나다고객님. 저는 치매진단 서비스를 제공하는 세븐포인트원 에이아이음성이에요. 오늘 하루는 어떠셨나요?'
        print("Bot: " + start_input)

    elif n == 6:
        break
    else:
        pass

    audiodata = get_speech()
    wav_data = librosa.util.buf_to_float(audiodata)

    # Transform to input
    feature = parser(wav_data)
    input_length = torch.LongTensor([len(feature)])
    STT_model = torch.load(STT_model_path, map_location=lambda storage, loc: storage).to(device)

    if isinstance(STT_model, nn.DataParallel):
        STT_model = STT_model.module

    y_hats = STT_model.recognize(feature.unsqueeze(0), input_length)
    sentence = STT_vocab.label_to_string(y_hats.cpu().detach().numpy())

    print(">> User:", revise(sentence))

    '''
    if user_input == 'bye':
        break
    '''
    # bot_input = input(">> Bot:")

    text_idx = tokenizer.encode(revise(sentence) + tokenizer.eos_token, return_tensors='pt')
    for i in range(len(generated_responses) - 1, len(generated_responses) - 3, -1):
        if i < 0:
            break
        encoded_vector = tokenizer.encode(generated_responses[i] + tokenizer.eos_token, return_tensors='pt')
        if text_idx.shape[-1] + encoded_vector.shape[-1] < 1000:
            text_idx = torch.cat([encoded_vector, text_idx], dim=-1)
        else:
            break
        encoded_vector = tokenizer.encode(past_user_inputs[i] + tokenizer.eos_token, return_tensors='pt')
        if text_idx.shape[-1] + encoded_vector.shape[-1] < 1000:
            text_idx = torch.cat([encoded_vector, text_idx], dim=-1)
        else:
            break
    text_idx = text_idx.to(device)
    inference_output = CH_model.generate(
        text_idx,
        max_length=1000,
        num_beams=5,
        top_k=20,
        no_repeat_ngram_size=4,
        length_penalty=0.65,
        repetition_penalty=2.0,
    )
    inference_output = inference_output.tolist()
    bot_response = tokenizer.decode(inference_output[0][text_idx.shape[-1]:], skip_special_tokens=True)
    print(" Bot:"+bot_response)
    
    past_user_inputs.append(revise(sentence))
    generated_responses.append(bot_response)

m = 0
while m < 2:
    m += 1
    if m == 1:
        start_exp = '에이아이 비대면 치매진단 알츠윈을 시작하겠습니다'
        print("Bot: " + start_exp)

inference_tacotron1([bot_response]) 

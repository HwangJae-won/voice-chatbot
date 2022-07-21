import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
matplotlib.use('Agg')

def plot_alignment(alignment, path, text):
    font_name = fm.FontProperties(fname="./TTS_tacotron1/util/malgun.ttf").get_name()
    matplotlib.rc('font', family=font_name, size=14)

    text = text.rstrip('_').rstrip('~')
    alignment = alignment[:len(text)]
    _, ax = plt.subplots(figsize=(len(text)/3, 5))
    ax.imshow(np.transpose(alignment), aspect='auto')#, origin='lower')
    plt.xlabel('Encoder timestep')
    plt.ylabel('Decoder timestep')
    text = [x if x != ' ' else '' for x in list(text)]
    plt.xticks(range(len(text)), text)
    plt.tight_layout()
    plt.savefig(path, format='png')

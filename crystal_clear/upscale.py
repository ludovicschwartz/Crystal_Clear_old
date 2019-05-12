from pydub import AudioSegment
import numpy as np
import scipy
from scipy.signal import stft
import torch.nn as nn
from fastai.basic_train import load_learner
from fastai.vision.image import open_image
from crystal_clear.prepare import make_3D_into_square, make_into_3D
from pathlib import Path
from tqdm import tqdm
from crystal_clear.utils import FeatureLoss
from fastai.basic_data import DatasetType
from fastai.vision.data import imagenet_stats, normalize, denormalize
import torch

mean_img_net = torch.tensor(imagenet_stats[0])
std_img_net = torch.tensor(imagenet_stats[1])


def predict_method(data, model, proc, bs, use_gpu, show_progress=True):
    use_gpu = torch.cuda.is_available() & use_gpu
    if use_gpu:
        print('gpu is used for prediction')
    else:
        print('cpu is used for prediction')
    preds = []
    n = data.shape[0]
    n_step = n // bs + (n % bs != 0)
    data, d_in = proc.f_in(data)
    for k in tqdm(range(n_step), disable=not show_progress):
        images, b_in = proc.f_batch_in(data[k*bs:min((k+1)*bs, n)])
        if use_gpu:
            images = images.cuda()
        preds.append(proc.f_batch_out(model(images).cpu().detach(),
                                      b_in))
    data = proc.f_out(torch.cat(preds, dim=0), d_in)
    return data


def upscale(path_file, proc, song_proc, learner=None, bs=32, use_gpu=True,
            show_progress=True):
    '''
    Upscale the file located in path_file using the learner. Warning, use_gpu
    must be coherent with the learner. If the learner is on CPU and cuda is
    available and use_gpu is set to True, an error will be raised
    '''
    song = AudioSegment.from_file(path_file)
    if learner is None:
        return song
    data_list = song_proc.process_in(song)
    data_list = [predict_method(data, learner.model, proc, bs, use_gpu)
                 for data in data_list]
    return song_proc.process_out(data_list)

from crystal_clear.utils import Processor, Song_Processor
from crystal_clear.utils import make_weighted_average_on_window
from crystal_clear.prepare import make_3D_into_square, make_into_3D
from fastai.vision.data import imagenet_stats, normalize, denormalize
import torch
import numpy as np


mean, std = (torch.Tensor(x) for x in imagenet_stats)


def f_in(data):
    m = data.max(axis=(1, 2, 3), keepdims=True)
    data = data / m
    data[np.isnan(data)] = 1
    return normalize(torch.Tensor(data), mean, std), m


def f_out(data, m):
    return denormalize(data, mean, std).numpy() * m

Image_proc = Processor(f_in, f_out)


def data_from_fourrier(Zxx):
    data, r = make_3D_into_square(make_into_3D(np.abs(Zxx)))
    data = np.stack(data)
    return [data], r


def fourrier_from_data(data_list, r, n_channels, sample_rate, f, t, Zxx):
    data = data_list[0]
    data = np.concatenate(list(data), axis=2)
    if r != 0:
        data = data[:, :, :(r-f.size)]
    data = data[:n_channels]
    angle = (Zxx / np.abs(Zxx))
    angle[np.isnan(angle)] = 1
    return angle * data


def data_from_fourrier2(Zxx):
    x = make_into_3D(np.abs(Zxx))
    nc, sz, n = x.shape
    x = np.pad(x, ((0, 0), (0, 0), (sz-1, sz-1)), mode='edge')
    x = np.stack([x[:, :, i:i+sz] for i in range(n+sz-1)])
    return [x], None


def fourrier_from_data2(data_list, r, n_channels, sample_rate, f, t, Zxx):
    data = data_list[0]
    data = make_weighted_average_on_window(data)
    data = data[:n_channels]
    angle = (Zxx / np.abs(Zxx))
    angle[np.isnan(angle)] = 1
    return angle * data
    
Image_song_proc = Song_Processor(data_from_fourrier, fourrier_from_data)
Image_song_proc2 = Song_Processor(data_from_fourrier2, fourrier_from_data2)

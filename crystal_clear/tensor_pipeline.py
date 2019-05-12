from crystal_clear.prepare import make_3D_into_square
from crystal_clear.utils import Processor, Song_Processor
from crystal_clear.utils import make_weighted_average_on_window
from fastai.vision.data import ImageList, ImageImageList
from fastai.vision.image import Image
from fastai.vision.data import normalize, denormalize
from functools import partial
import torch
import numpy as np
from tqdm import tqdm


class TensorImageList(ImageList):
    def open(self, fn):
        return Image(torch.load(fn, map_location='cpu').type(torch.float))


class TensorImageImageList(ImageImageList):
    def open(self, fn):
        return Image(torch.load(fn, map_location='cpu').type(torch.float))


def calculate_data_stats(path, meta):
    '''
    Calculate and saves the mean and standard deviation of all the tensor
    saved in path / 'crap_tensor' whose name is in meta.
    '''
    mean = 0
    mean_sq = 0
    n_row = meta.shape[0]
    for name in tqdm(meta.spectr_id):
        x = torch.load(path / 'crap_tensor' / name,
                       map_location='cpu').type(torch.float)
        mean += x.mean(dim=[1, 2])
        mean_sq += x.pow(2).mean(dim=[1, 2])
    mean, mean_sq = mean / n_row, mean_sq / n_row
    std = (mean_sq - mean.pow(2)).sqrt()
    torch.save((mean, std), path / 'data_stats.pkl')


def create_tensor_Processor(data_stats):
    mean, std = data_stats

    def f_in(data):
        return normalize(torch.Tensor(data), mean, std), None

    def f_out(data, _):
        return denormalize(data, mean, std).numpy()
    return Processor(f_in, f_out)


def data_from_fourrier(Zxx):
    channels = Zxx.shape[0]
    data = []
    for i in range(channels):
        x = np.stack((np.real(Zxx[i]),
                      np.imag(Zxx[i]),
                      np.abs(Zxx[i])))
        x, r = make_3D_into_square(x)
        data.append(np.stack(x))
    return data, r


def fourrier_from_data(data_list, r, n_channels, sample_rate, f, t, Zxx, x):
    list_channels = []
    for data in data_list:
        data = np.concatenate(list(data), axis=2)
        if r != 0:
            data = data[:, :, :(r-f.size)]
        data = (data[0] + data[1] * 1j) * (data[2]**2
                                           / (data[0]**2
                                              + data[1]**2))**(x / 2)
        data[np.isnan(data)] = 0
        list_channels.append(data)
    data = np.stack(list_channels)
    return data


def data_from_fourrier2(Zxx):
    channels = Zxx.shape[0]
    data = []
    for i in range(channels):
        x = np.stack((np.real(Zxx[i]),
                      np.imag(Zxx[i]),
                      np.abs(Zxx[i])))
        nc, sz, n = x.shape
        x = np.pad(x, ((0, 0), (0, 0), (sz-1, sz-1)), mode='edge')
        x = np.stack([x[:, :, i:i+sz] for i in range(n+sz-1)])
        data.append(x)
    return data, None


def fourrier_from_data2(data_list, r, n_channels, sample_rate, f, t, Zxx, x):
    list_channels = []
    for data in data_list:
        data = make_weighted_average_on_window(data)
        data = (data[0] + data[1] * 1j) * (data[2]**2
                                           / (data[0]**2
                                              + data[1]**2))**(x / 2)
        data[np.isnan(data)] = 0
        list_channels.append(data)
    data = np.stack(list_channels)
    return data


def create_song_tensor_Processor(x=0.5):
    return Song_Processor(data_from_fourrier,
                          partial(fourrier_from_data, x=x))


def create_song_tensor_Processor2(x=0.5):
    return Song_Processor(data_from_fourrier2,
                          partial(fourrier_from_data2, x=x))

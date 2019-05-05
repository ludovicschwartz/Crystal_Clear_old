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


def upscale(path_file, learner=None, bs=32, use_gpu=True, show_progress=True):
    '''
    Upscale the file located in path_file using the learner.
    '''
    use_gpu = torch.cuda.is_available() & use_gpu
    if use_gpu:
        print('gpu is used for prediction')
    else:
        print('cpu is used for prediction')
    nperseg = 254
    song = AudioSegment.from_file(path_file)
    if learner is None:
        return song
    n_channels = song.channels
    samples = np.array(song.get_array_of_samples()).reshape(n_channels,
                                                            -1, order='F')
    # return samples
    sample_rate = song.frame_rate
    n_frames = samples.shape[1]
    f, t, Zxx = scipy.signal.stft(samples, sample_rate, nperseg=nperseg)
    data, r = make_3D_into_square(make_into_3D(np.abs(Zxx)))
    data = np.stack(data)
    m = data.max(axis=(1, 2, 3), keepdims=True)
    data = data / m
    data[np.isnan(data)] = 1
    preds = []
    n_step = data.shape[0] // bs + (data.shape[0] % bs != 0)
    for k in tqdm(range(n_step), disable=not show_progress):
        if use_gpu:
            images = normalize(torch.tensor(data[k*bs:min((k+1)*bs,
                                                          data.shape[0])]),
                               mean_img_net,
                               std_img_net).cuda()
        else:
            images = normalize(torch.tensor(data[k*bs:min((k+1)*bs,
                                                          data.shape[0])]),
                               mean_img_net,
                               std_img_net)
        preds.append(denormalize(learner.model(images).cpu().detach(),
                                 mean_img_net,
                                 std_img_net).numpy())
    data = np.concatenate(preds, axis=0) * m
    data = np.concatenate(list(data), axis=2)
    if r != 0:
        data = data[:, :, :(r-f.size)]
    data = data[:n_channels]
    angle = (Zxx / np.abs(Zxx))
    angle[np.isnan(angle)] = 1
    new_Zxx = angle * data
    _, reconstructed = scipy.signal.istft(new_Zxx, sample_rate)
    reconstructed = reconstructed[:, :n_frames]
    reconstructed = np.rint(reconstructed).astype(np.int16)
    song._data = bytearray(reconstructed.reshape(1, -1, order='F'))
    return song


def make_weighted_average_on_window(array, predict_method, padded=True):
    nc, sz, n = array.shape
    if not padded:
        array = np.pad(array, ((0, 0), (0, 0), (sz-1, sz-1)), mode='edge')
    else:
        n = n - 2 * (sz-1)
    sliding = predict_method(np.stack([array[:, :, i:i+sz]
                        for i in range(n+sz-1)]))

    sliding_frame = np.lib.stride_tricks.as_strided(
        # Reverse last dimension since we want the last column from the first
        # frame.
        # Need to copy again because `[::-1]` creates a view with negative
        # stride, but we want actual reversal to work with the strides below.
        # (There's perhaps a smart way of adjusting the strides below in order
        # to not make a copy here.)
        sliding[:, :, :, ::-1].copy(),
        # Second dimension corresponds to the 128 consecutive frames.
        # Previous last dimension is dropped since we're selecting the
        # column that corresponds to the current n-position
        shape=(sz, n, nc, sz),
        # First dimension (frame position): Move one frame and one column ahead
        # (actually want to move one column less in `sliding` but since we
        # reverted order of columns we need to move one ahead now)
        # -> move across first dimension of `sliding` + last dimension of
        # `sliding`.
        # Second dimension (n-position): Moving one frame ahead
        # -> move across first dimension of `sliding`.
        # Remaining two dimensions:
        # Move within frames (channel and row dimensions).
        strides=((sliding.strides[0] + sliding.strides[-1],)
                 + sliding.strides[:1] + sliding.strides[1:3])
    )
    # Weights are independent of the n-position -> we can precompute.
    parity = [] if sz % 2 == 0 else [sz//2]
    weights = 1 + np.concatenate([np.arange(sz//2), parity,
                                  np.arange(sz//2)[::-1]])
    weights = weights.astype(float) / weights.sum()
    weights = weights[:, None, None, None]  # Prepare for broadcasting.
    return np.moveaxis(np.sum(sliding_frame * weights, axis=0), 0, 2)


def upscale2(path_file, learner=None, bs=32, use_gpu=True, show_progress=True):
    '''
    Upscale the file located in path_file using the learner.
    '''
    if use_gpu:
        print('gpu is used for prediction')
    else:
        print('cpu is used for prediction')
    nperseg = 254
    sz = nperseg // 2 + 1
    n_pad = sz ** 2
    song = AudioSegment.from_file(path_file)
    if learner is None:
        return song
    from_img = False
    n_channels = song.channels
    samples = np.array(song.get_array_of_samples()).reshape(n_channels,
                                                            -1, order='F')
    n_frames = samples.shape[1]
    # return samples
    sample_rate = song.frame_rate
    # We 0-pad the edge of the music
    samples = np.pad(samples, ((0, 0), (n_pad, n_pad)), mode='edge')
    f, t, Zxx = scipy.signal.stft(samples, sample_rate, nperseg=nperseg)
    t = t[sz-1:-sz+1]
    data = make_into_3D(np.abs(Zxx))
    Zxx = Zxx[:, :, sz-1:-sz+1]
    n = data.shape[-1]

    def predict_method(data):
        m = data.max(axis=(1, 2, 3), keepdims=True)
        data = data / m
        data[np.isnan(data)] = 1
        preds = []
        n_step = data.shape[0] // bs + (data.shape[0] % bs != 0)
        for k in tqdm(range(n_step), disable=not show_progress):
            if use_gpu:
                images = normalize(torch.tensor(data[k*bs:min((k+1)*bs,
                                                              data.shape[0])]),
                                   mean_img_net,
                                   std_img_net).cuda()
            else:
                images = normalize(torch.tensor(data[k*bs:min((k+1)*bs,
                                                              data.shape[0])]),
                                   mean_img_net,
                                   std_img_net)
            preds.append(denormalize(learner.model(images).cpu().detach(),
                                     mean_img_net,
                                     std_img_net).numpy())
        data = np.concatenate(preds, axis=0) * m
        return data
    data = make_weighted_average_on_window(data, predict_method)
    # Same as upscale here
    data = data[:n_channels]
    angle = (Zxx / np.abs(Zxx))
    angle[np.isnan(angle)] = 1
    new_Zxx = angle * data
    _, reconstructed = scipy.signal.istft(new_Zxx, sample_rate)
    reconstructed = reconstructed[:, :n_frames]
    reconstructed = np.rint(reconstructed).astype(np.int16)
    song._data = bytearray(reconstructed.reshape(1, -1, order='F'))
    return song
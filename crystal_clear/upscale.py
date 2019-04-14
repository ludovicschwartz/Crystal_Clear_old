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


def upscale(path_file, learner=None):
    '''
    Upscale the file located in path_file using the learner.
    '''
    nperseg = 254
    song = AudioSegment.from_file(path_file)
    if learner is None:
        return song
    from_img = False
    n_channels = song.channels
    samples = np.array(song.get_array_of_samples()).reshape(n_channels,
                                                            -1, order='F')
    # return samples
    sample_rate = song.frame_rate
    f, t, Zxx = scipy.signal.stft(samples, sample_rate, nperseg=nperseg)
    data, r = make_3D_into_square(make_into_3D(np.abs(Zxx)))
    n_window = len(data)
    for i, array in tqdm(enumerate(data), total=n_window):
        m = array.max()
        array = array / m
        data[i] = learner.predict(array)[0].data.numpy().astype(np.float32) * m
    if r != 0:
        data[-1] = data[-1][:, :, :r]
    data = np.concatenate(data, axis=2)
    data = data[:n_channels]
    angle = (Zxx / np.abs(Zxx))
    angle[np.isnan(angle)] = 1
    new_Zxx = angle * data
    _, reconstructed = scipy.signal.istft(new_Zxx, sample_rate)
    reconstructed = np.rint(reconstructed).astype(np.int16)
    song._data = bytearray(reconstructed.reshape(1, -1, order='F'))
    return song


def make_weighted_average_on_window(array, predict_method, padded=True):
    nc, sz, n = array.shape
    if not padded:
        array = np.pad(array, ((0, 0), (0, 0), (sz-1, sz-1)), mode='edge')
    else:
        n = n - 2 * (sz-1)
    sliding = np.stack([predict_method(array[:, :, i:i+sz])
                        for i in tqdm(range(n+sz-1))])

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


def upscale2(path_file, learner):
    '''
    Upscale the file located in path_file using the learner.
    '''
    nperseg = 254
    sz = nperseg // 2 + 1
    song = AudioSegment.from_file(path_file)
    if learner is None:
        return song
    from_img = False
    n_channels = song.channels
    samples = np.array(song.get_array_of_samples()).reshape(n_channels,
                                                            -1, order='F')
    # return samples
    sample_rate = song.frame_rate
    # We 0-pad the edge of the music
    samples = np.pad(samples, ((0, 0), (sz-1, sz-1)), mode='edge')
    f, t, Zxx = scipy.signal.stft(samples, sample_rate, nperseg=nperseg)
    t = t[sz-1:-sz+1]
    data = make_into_3D(np.abs(Zxx))
    Zxx = Zxx[:, :, sz-1:-sz+1]
    n = data.shape[-1]

    def predict_method(a):
        m = a.max()
        return learner.predict(a / m)[0].data.numpy().astype(np.float32) * m
    data = make_weighted_average_on_window(data, predict_method)
    # Same as upscale here
    data = data[:n_channels]
    angle = (Zxx / np.abs(Zxx))
    angle[np.isnan(angle)] = 1
    new_Zxx = angle * data
    _, reconstructed = scipy.signal.istft(new_Zxx, sample_rate)
    reconstructed = np.rint(reconstructed).astype(np.int16)
    song._data = bytearray(reconstructed.reshape(1, -1, order='F'))
    return song


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input, target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()

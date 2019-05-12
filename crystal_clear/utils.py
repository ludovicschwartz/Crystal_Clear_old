from torch import nn
from fastai.core import ifnone
import scipy
from scipy.signal import stft
import numpy as np
import torch


def make_weighted_average_on_window(sliding):
    n, nc, sz, _ = sliding.shape
    n = n - (sz-1)
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
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()


def wrap_for_processor(f, out=True):
    if out:
        return lambda x, y: f(x)
    else:
        return lambda x: (f(x), None)


class Processor():
    '''
    This class is used to process the data before and after
    putting it in the model. f_in will be applied first, then f_batch_in will
    be applied at every batch and f_batch_out at the output of every batch and
    f_out at the complete output. f_in and f_batch_in must return a tuple with
    the first element being the output and the second element can be None or
    information that is needed for f_out and f_batch out. Similarly f_out and
    f_batch out must take tuples as input, the first element being the output
    of the model and the second element coming from f_in or f_batch_in.
    '''
    def __init__(self, f_in=None, f_out=None, f_batch_in=None,
                 f_batch_out=None):
        self.f_in = ifnone(f_in, lambda x: (torch.Tensor(x), None))
        self.f_out = ifnone(f_out, lambda x, y: x.numpy())
        self.f_batch_in = ifnone(f_batch_in, lambda x: (x, None))
        self.f_batch_out = ifnone(f_batch_out, lambda x, y: x)


nperseg = 254


class Song_Processor():
    def __init__(self, data_from_fourrier, fourrier_from_data):
        self.song_info = False
        self.data_from_fourrier = data_from_fourrier
        self.fourrier_from_data = fourrier_from_data

    def process_in(self, song):
        self.channels = song.channels
        samples = np.array(song.get_array_of_samples()).reshape(self.channels,
                                                                -1, order='F')
        self.sample_rate = song.frame_rate
        self.n_frames = samples.shape[1]
        f, t, Zxx = scipy.signal.stft(samples, self.sample_rate,
                                      nperseg=nperseg)
        self.f, self.t, self.Zxx = f, t, Zxx
        self.song = song
        data_list, self.f_info = self.data_from_fourrier(Zxx)
        self.song_info = True
        return data_list

    def process_out(self, data_list):
        if not self.song_info:
            raise Exception('No song_info can be found, process_in must\
                             be called before calling process_out')
        new_Zxx = self.fourrier_from_data(data_list,
                                          self.f_info,
                                          self.channels,
                                          self.sample_rate,
                                          self.f, self.t, self.Zxx)
        _, reconstructed = scipy.signal.istft(new_Zxx, self.sample_rate)
        reconstructed = reconstructed[:, :self.n_frames]
        reconstructed = np.rint(reconstructed).astype(np.int16)
        self.song._data = bytearray(reconstructed.reshape(1, -1, order='F'))
        song = self.song
        self.song_info = False
        for x in self.__dict__.copy():
            if x not in ('song_info', 'data_from_fourrier',
                         'fourrier_from_data'):
                delattr(self, x)
        return song




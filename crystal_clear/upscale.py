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


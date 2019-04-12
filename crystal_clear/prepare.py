import numpy as np
from pydub import AudioSegment
from pathlib import Path
from PIL import Image
import os


def path_mp3(track_id, folder='fma_small'):
    track_id = str(track_id)
    track_id = '0' * (6-len(track_id)) + track_id
    return Path('./data/') / folder / track_id[:3] / (track_id + '.mp3')


def open_random_file(df_meta):
    sample_meta = df_meta.sample().iloc[0]
    path = path_mp3(sample_meta.track_id, f'fma_{sample_meta.set_subset}')
    song = AudioSegment.from_mp3(path)
    return(song)


def make_into_3D(data):
    '''
    Transform an array of shape (1, x, y), (2, x, y) or (x, y) into an array of
    shape (3, x, y). If the array has 1 channel, copy it two times to fill the
    two others channel. If it has two channels, fills the last channel with the
    mean of other channels
    '''
    if len(data.shape) == 2:
        return np.stack([data for i in range(3)])
    c, nf, nt = data.shape
    if c == 2:
        data_avg = np.mean([data[0], data[1]], axis=0)
        return np.stack((data[0], data[1], data_avg))
    elif c == 1:
        return make_into_3D(data[0])
    else:
        print('Array shape not understood.')


def make_3D_into_square(data):
    '''
    Transform an array of shape (3, x, y) into a sequence of array of shape
    (3, x, x). It will pad the last array with zeros. And it will return a
    tuple, the first being the list of array, and the second, the rest of the
    division of y by x.
    '''
    c, nf, nt = data.shape
    l = [data[:, :, i*nf:(i+1)*nf] for i in range(nt // nf)]
    r = nt % nf
    if r != 0:
        l.append(np.pad(data[:, :, nf*(nt // nf):],
                        ((0, 0), (0, 0), (0, (nf-r))),
                        mode='constant'))
    return(l, r)


def save_to_image(list_square, name, folder):
    folder = Path(folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i, spectr in enumerate(list_square):
        spectr = (spectr * 255 / np.max(spectr)).astype('uint8')
        Image.fromarray(np.moveaxis(spectr,
                                    0, -1)).save(folder / f'{name}_{i}.png')


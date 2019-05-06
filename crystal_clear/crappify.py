from pydub import AudioSegment
from crystal_clear.prepare import *
import os
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import scipy
from scipy.signal import stft



def crappify_song(path1, path2, **kwargs):
    song = AudioSegment.from_mp3(path1)
    song.export(path2, **kwargs)


def crappify_group(df_meta, new_path, overwrite=False,
                   df_valid=None, **kwargs):
    if df_valid is not None:
        if 'track_id' not in df_valid.columns \
           or 'subset' not in df_valid.columns:
            print('''
                 Error in df_valid, if specified, df_valid must have a
                 'track_id' and a 'subset' column.
                 ''')
            return
        else:
            df_valid = df_valid[['track_id', 'subset']]
    new_path = Path(new_path)
    df_meta = df_meta.copy()
    if not os.path.exists(new_path / 'mp3'):
        os.makedirs(new_path / 'mp3')
    if not os.path.exists(new_path / 'meta'):
        os.makedirs(new_path / 'meta')
    n_rows = df_meta.shape[0]
    for index, row in tqdm(df_meta.iterrows(), total=n_rows):
        if not overwrite and os.path.exists(new_path / f'{row.track_id}.mp3'):
            continue
        subset_folder = f'fma_{row.set_subset}'
        path = path_mp3(row.track_id, subset_folder)
        try:
            crappify_song(path, new_path / 'mp3' / f'{row.track_id}.mp3',
                          **kwargs)
        except:
            print(f'error on track {row.track_id}')
            df_meta.drop(index, inplace=True)
    if df_valid is not None:
        df_meta = df_meta.merge(df_valid, on='track_id', how='left')
        if df_meta.subset.isna().sum() > 0:
            print('''
                  There are some 'track_id' not contained in df_valid. Their
                  'subset' has been set to 'unspecified' 
                  ''')
        df_meta.subset = df_meta.subset.fillna('unspecified')
    else:
        np.random.seed(42)
        n_rows = df_meta.shape[0]
        valid_set = np.random.rand(n_rows) > 0.8
        valid = np.where(valid_set, 'valid', 'train')
        df_meta['subset'] = valid
    df_meta.reset_index(drop=True).to_csv(new_path / 'meta' / 'meta.csv',
                                          index=None)


def to_list_square(path):
    song = AudioSegment.from_mp3(path)
    samples = np.array(song.get_array_of_samples()).reshape(song.channels, -1,
                                                            order='F')
    sample_rate = song.frame_rate
    f, t, Zxx = scipy.signal.stft(samples, sample_rate, nperseg=254)
    return make_3D_into_square(make_into_3D(np.abs(Zxx))), song.channels


def to_list_square2(path):
    song = AudioSegment.from_mp3(path)
    channels = song.channels
    samples = np.array(song.get_array_of_samples()).reshape(channels, -1,
                                                            order='F')
    sample_rate = song.frame_rate
    f, t, Zxx = scipy.signal.stft(samples, sample_rate, nperseg=254)
    for i in range(channels):
        data = np.stack((np.real(Zxx[i]),
                         np.imag(Zxx[i]),
                         np.abs(Zxx[i])))
        if i == 0:
            x, r = make_3D_into_square(data)
        if i == 1:
            x = [x, make_3D_into_square(data)[0]]
    return x, r, channels


def create_images(meta, path_data):
    path_data = Path(path_data)
    n_rows = meta.shape[0]
    list_id = []
    list_n_window = []
    list_rest = []
    valid = meta.subset
    genre = meta.track_genre_top
    list_n_channels = []
    for _, row in tqdm(meta.iterrows(), total=n_rows):
        track_id = row.track_id
        subset_folder = f'fma_{row.set_subset}'
        list_square, rest, channels = to_list_square(path_data /
                                                     f'mp3/{track_id}.mp3')
        list_id.append(track_id)
        list_n_window.append(len(list_square))
        list_n_channels.append(channels)
        list_rest.append(rest)
        save_to_image(list_square, track_id, path_data / 'crap_spectr/')
        path_orig = path_mp3(track_id, folder=subset_folder)
        list_square, _, _ = to_list_square(path_orig)
        save_to_image(list_square, track_id, path_data / 'orig_spectr/')
    pd.DataFrame({'track_id': list_id, 'n_window': list_n_window,
                  'rest': list_rest, 'genre': genre,
                  'n_channels': list_n_channels,
                  'subset': valid}).to_csv(path_data / 'meta/meta_mp3.csv',
                                           index=None)


def create_tensor(meta, path_data):
    path_data = Path(path_data)
    n_rows = meta.shape[0]
    list_id = []
    list_n_window = []
    list_rest = []
    valid = meta.subset
    genre = meta.track_genre_top
    list_n_channels = []
    for _, row in tqdm(meta.iterrows(), total=n_rows):
        track_id = row.track_id
        subset_folder = f'fma_{row.set_subset}'
        list_square, rest, channels = to_list_square2(path_data /
                                                      f'mp3/{track_id}.mp3')
        list_id.append(track_id)
        if channels == 1:
            list_n_window.append(len(list_square))
        else:
            list_n_window.append(len(list_square[0]))
        list_n_channels.append(channels)
        list_rest.append(rest)
        is_mono = channels == 1
        save_to_tensor(list_square, track_id, path_data / 'crap_tensor/',
                       is_mono=is_mono)
        path_orig = path_mp3(track_id, folder=subset_folder)
        list_square, _, _ = to_list_square2(path_orig)
        save_to_tensor(list_square, track_id, path_data / 'orig_tensor/',
                       is_mono=is_mono)
    pd.DataFrame({'track_id': list_id, 'n_window': list_n_window,
                  'rest': list_rest, 'genre': genre,
                  'n_channels': list_n_channels,
                  'subset': valid}).to_csv(path_data / 'meta/meta_mp3.csv',
                                           index=None)
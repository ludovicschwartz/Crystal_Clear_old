'''
This files contains everything needed to deal with the medatata.csv files.
'''
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

dirname = Path(__file__).parent


def clean_meta():
    path_to_meta = Path(os.path.join(dirname, '../data/fma_metadata'))
    if os.path.exists(path_to_meta / 'tracks_clean.csv'):
        return pd.read_csv(path_to_meta / 'tracks_clean.csv')

    df_meta = pd.read_csv(path_to_meta / 'tracks.csv', header=None)

    def combine_list(list_1, list_2):
        l = []
        for i in range(len(list_1)):
            l.append(list_1[i] + '_' + list_2[i])
        return(l)

    columns = ['track_id'] + combine_list(df_meta.iloc[0, 1:].values,
                                          df_meta.iloc[1, 1:].values)
    df_meta = df_meta.iloc[3:, :]
    df_meta.columns = columns
    df_meta.to_csv(path_to_meta / 'tracks_clean.csv', index=None)
    return(df_meta.reset_index(drop=True))


def small_meta():
    df_meta = clean_meta()
    return df_meta[df_meta.set_subset == 'small'].reset_index(drop=True)


def create_meta_spectr(meta_sound, orig_meta, meta_path):
    '''
    Meta_sound is a dataframe that countains 3 columns, track_id, n_window and
    rest. Rest is needed to reconstruct the sound file, but we only need
    n_window and track_id to get the info on the files.
    '''
    meta_path = Path(meta_path)
    meta_full = meta_sound[['track_id',
                            'n_window']].merge(orig_meta, on='track_id')[['track_id',
                                                                          'n_window',
                                                                          'track_genre_top',
                                                                          'subset']]
    n_rows = meta_full.shape[0]
    list_id = []
    list_genre = []
    list_valid = []
    for i, row in tqdm(meta_full.iterrows(), total=n_rows):
        for j in range(row.n_window):
            list_id.append(f'{row.track_id}_{j}.png')
            list_genre.append(row.track_genre_top)
            list_valid.append(row.subset)
    df = pd.DataFrame({'spectr_id': list_id, 'genre': list_genre,
                       'subset': list_valid})
    df.to_csv(meta_path / 'meta_spectr.csv', index=None)


def create_meta_tensor(meta_sound, orig_meta, meta_path):
    '''
    Meta_sound is a dataframe that countains 3 columns, track_id, n_window and
    rest. Rest is needed to reconstruct the sound file, but we only need
    n_window and track_id to get the info on the files.
    '''
    meta_path = Path(meta_path)
    meta_full = meta_sound[['track_id',
                            'n_window',
                            'n_channels']].merge(orig_meta,
                                                 on='track_id')[['track_id',
                                                                 'n_window',
                                                                 'track_genre_top',
                                                                 'n_channels',
                                                                 'subset']]
    n_rows = meta_full.shape[0]
    list_id = []
    list_genre = []
    list_valid = []
    for i, row in tqdm(meta_full.iterrows(), total=n_rows):
        for j in range(row.n_window):
            if row.n_channels == 1:
                c_name = ['']
            else:
                c_name = ['_l', '_r']
            for n in c_name:
                list_id.append(f'{row.track_id}{n}_{j}.ti')
                list_genre.append(row.track_genre_top)
                list_valid.append(row.subset)
    df = pd.DataFrame({'spectr_id': list_id, 'genre': list_genre,
                       'subset': list_valid})
    df.to_csv(meta_path / 'meta_tensor.csv', index=None)
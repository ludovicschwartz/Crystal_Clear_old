from pydub import AudioSegment
from crystal_clear.prepare import path_mp3
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np


def crappify_song(path1, path2, **kwargs):
    song = AudioSegment.from_mp3(path1)
    song.export(path2, **kwargs)


def crappify_group(df_meta, new_path, overwrite=False, **kwargs):
    new_path = Path(new_path)
    df_meta = df_meta.copy()
    if not os.path.exists(new_path):
        os.makedirs(new_path)
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
    np.random.seed(42)
    n_rows = df_meta.shape[0]
    valid_set = np.random.rand(n_rows) > 0.8
    valid = np.where(valid_set, 'valid', 'train')
    df_meta['subset'] = valid
    df_meta.reset_index(drop=True).to_csv(new_path / 'meta' / 'meta.csv',
                                          index=None)

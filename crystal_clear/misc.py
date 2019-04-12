def reconstruct(path_mp3, path_images, track_id, learner=None):
    nperseg = 254
    song = AudioSegment.from_mp3(path_mp3)
    n_channels = song.channels
    samples = np.array(song.get_array_of_samples()).reshape(n_channels, -1, order='F')
    # return samples
    sample_rate = song.frame_rate
    f, t, Zxx = scipy.signal.stft(samples, sample_rate, nperseg=nperseg)
    data, r = make_3D_into_square(make_into_3D(np.abs(Zxx)))
    n_window = len(data)
    for i, array in enumerate(data):
        m = array.max()
        if learner is None:
            data_from_img = np.array(Image.open(path_images / f'{track_id}_{i}.png'))
            data_from_img = np.moveaxis(data_from_img, -1, 0)
            data_from_img = data_from_img / 255 * m
        else:
            data_from_img = learner.predict(open_image(path_images / f'{track_id}_{i}.png'))[0].data.numpy()
            data_from_img = data_from_img * m
        data[i] = data_from_img.astype(np.float32)
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
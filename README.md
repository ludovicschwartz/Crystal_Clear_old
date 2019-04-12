# Crystal_Clear
Deep Learning technology to upscale music.
The Dataset used to train the models is the FMA(Free music Archive), a library of high-quality, leagal audio downloads.
More specifically, I used https://github.com/mdeff/fma . They did a great job and provided an easy to use and very rich Dataset.
@inproceedings{fma_dataset,
  title = {FMA: A Dataset for Music Analysis},
  author = {Defferrard, Micha\"el and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},
  booktitle = {18th International Society for Music Information Retrieval Conference},
  year = {2017},
  url = {https://arxiv.org/abs/1612.01840},
}

The pipeline is the following : 

- Take as input an audio file
- Generate the stft of this file
- Take the spectrogram as an img and cut it in square image
- Pass those images to the model (a Unet) which will upscale them
- Update the stft of the audio file with the new spectrogram
- Create the reconstructed audio file with inverse stft



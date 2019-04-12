# Crystal_Clear
Deep Learning technology to upscale music.
The Dataset used to train the models is the FMA(Free music Archive), a library of high-quality, leagal audio downloads.
More specifically, I used https://github.com/mdeff/fma . They did a great job and provided an easy to use and very rich Dataset.  

  title = {FMA: A Dataset for Music Analysis},  
  author = {Defferrard, Micha\"el and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},  
  booktitle = {18th International Society for Music Information Retrieval Conference},  
  year = {2017},  
  url = {https://arxiv.org/abs/1612.01840}  

The pipeline is the following : 

- Take as input an audio file  
- Generate the stft of this file  
- Take the spectrogram as an img and cut it in square image  
- Pass those images to the model (a Unet) which will upscale them  
- Update the stft of the audio file with the new spectrogram  
- Create the reconstructed audio file with inverse stft  

To use this repo :

- Create an environment with the environment.yml file  
- To listen to existing samples, you can use the listening part of the sample.ipynb notebook  
- To apply the model on new_data, use the upscale.ipynb notebook. It takes around 2 seconds for 3 seconds of music on my CPU   (intel i7) or 1 sec for 30 sec of musics on my GPU (nvidia 1070).  

I'll add instructions to train the model.



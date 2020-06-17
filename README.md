# Crystal_Clear
Deep Learning technology to upscale music.
The Dataset used to train the models is the FMA(Free music Archive), a library of high-quality, leagal audio downloads.
More specifically, I used https://github.com/mdeff/fma . They did a great job and provided an easy to use and very rich Dataset.  

  title = {FMA: A Dataset for Music Analysis},  
  author = {Defferrard, Micha\"el and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},  
  booktitle = {18th International Society for Music Information Retrieval Conference},  
  year = {2017},  
  url = {https://arxiv.org/abs/1612.01840}  

The latest model is available on a webapp on the following link : https://crystalclear.staraurryon.com/ (The link is currently not unoperational and shall be fixed soon).
Big thanks to my brother [StarAurryon](https://github.com/StarAurryon) for helping me on creating this webapp. Feel free to test the model and if you get interesting results, I would be thrilled to hear about them.

The pipeline is the following : 

- Take as input an audio file  
- Generate the stft of this file  
- Take the spectrogram as an img and cut it in square image
- There are two pipelines here called the `image_pipeline` and `tensor_pipeline` depending on how you perform this.
- Pass those images to the model (a Unet) which will upscale them  
- Update the stft of the audio file with the new spectrogram  
- Create the reconstructed audio file with inverse stft  

To use this repo :

- Create an environment with the environment.yml file  
- To listen to existing samples, you can use the listening part of the sample.ipynb notebook  
- To apply the model on new_data, use the upscale.ipynb notebook. It takes around 2 seconds for 3 seconds of music on my CPU   (intel i7) or 1 sec for 30 sec of musics on my GPU (nvidia 1070).  

Note : If you want to apply the model on new_data, the last model can be found on :
- https://drive.google.com/open?id=1w0H4W7ZCVJARe9rokfMCOOZn4yJXn3yY (exported version)
- https://drive.google.com/open?id=1FHqzGyEaqESOwbpKh3fYZF0cPTVaiaT_ (If you want to retrain)
- https://drive.google.com/open?id=18yXjNTXvhLmX0GqR2VYK1oJ2A1yUukuH (The target specific Vgg16 classifier weights used for the feature loss)

If you want to train on the data :
- Launch the Starter_kit.ipynb notebook
- Execute all cells to download and preprocess the data(It is kinda long at the moment, around 2-3 hours)
- Classification.ipynb and Vgg_16_class.ipynb contains code to train a genre classifier on the spectrogram and 





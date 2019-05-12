30/04/2019 :
- Added parallelization in image superresolution in both upscale methods. Those
methods are now much faster (2-3 times on my computer)

01/05/2019 :
- Created the CHANGES.md file to keep track of important changes.
- Created a notebook Starter_kit.ipynb to get the data and preprocess it. It is
now possible to experiment and train the models yourself.

06/05/2019 :
- Started working on the `tensor_pipeline`

07/05/2019 : 
- Added license to the repo

12/05/2019 :
- Added the tensor pipeline to the starter kit.
- Refactor of the training notebook of Vgg_16 to use fastai and to be compatible with both pipelines.
- Switch to pytorch 1.1 from pytorch 1.0 (updated the environment file)
- Refactor of the upscale scripts and notebook to work with both pipelines and be more customizable.

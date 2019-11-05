# Web Stereo Video Supervision for Depth Prediction from Dynamic Scenes
testing code for the paper [Wang etal. 3DV19](https://arxiv.org/pdf/1904.11112.pdf)
## Installation
tested with Python 3.6 with cuda 9.0 on Ubuntu16.04
### install dependicies:
- opencv-python
- numpy
- pytorch 0.4.1
### install flownet2.0
```
cd networks/flownet2
bash install.sh
```
## download model
- download [ckpts.tar](https://drive.google.com/open?id=1Zx2sU_3cnHT4okqbSdIncuWLpQKLuCpm)
- extract the content to ./ckpts/


## predict depth from 2 images
```
CUDA_VISIBLE_DEVICES=0 python demo.py demo/im1.jpg demo/im2.jpg
```

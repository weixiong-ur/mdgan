# Learning to Generate Time-lapse Videos Using Multi-stage Dynamic Generative Adversarial Networks

This is the official code of the CVPR 2018 PAPER.

[CVPR 2018 PAPER](https://arxiv.org/pdf/1709.07592.pdf) | [Project Page](https://sites.google.com/site/whluoimperial/mdgan) | [Dataset](https://drive.google.com/open?id=1xWLiU-MBGN7MrsFHQm4_yXmfHBsMbJQo)

## Usage

1. Requirements:
	* download our time-lapse [dataset](https://drive.google.com/open?id=1xWLiU-MBGN7MrsFHQm4_yXmfHBsMbJQo)
	* python2.7
	* pytorch 0.3.0 or 0.3.1
	* ffmpeg 
2. Testing:
	* download our [pretrained models](https://drive.google.com/open?id=1uP2lNPutWgKhRqbwxZFdhQ9LapENCQXk)
	* run `python test.py --cuda --testf your_test_dataset_folder`
3. Sample outputs:
	* in `./sample_outputs`there are mp4 files which are generated on my machine.

## Citing
```
@InProceedings{Xiong_2018_CVPR,
author = {Xiong, Wei and Luo, Wenhan and Ma, Lin and Liu, Wei and Luo, Jiebo},
title = {Learning to Generate Time-Lapse Videos Using Multi-Stage Dynamic
Generative Adversarial Networks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition
(CVPR)},
month = {June},
year = {2018}
}
```



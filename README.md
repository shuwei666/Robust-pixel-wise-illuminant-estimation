# Robust pixel-wise illuminant estimation algorithm for images with a low bit-depth
Shuwei Yue and *[Minchen Wei](https://www.polyucolorlab.com/)

*Color, Imaging, and Metaverse Research Center, The Hong Kong Polytechnic University.*

**Abstract:**
> Conventional illuminant estimation methods were developed for scenes with a uniform illumination, while recently developed methods, such as pixel-wise methods, estimate the illuminants at the pixel level, making them applicable to a wider range of scenes. It was found that the same pixel-wise algorithm had very different performance when applied to images with different bit-depths, with up to a 30% decrease in accuracy for images having a lower bit-depth. Image signal processing (ISP) pipelines, however, prefer to deal with images with a lower bit-depth. In this paper, the analyses show that such a reduction was due to the loss of details and increase of noises, which were never identified in the past. We propose a method combining the L1 loss optimization and physical-constrained post-processing. The proposed method was found to result in around 40% higher estimation accuracy, in comparison to the state-of-the-art DNN-based methods.

![Overview](https://github.com/shuwei666/Robust-pixel-wise-illuminant-estimation/assets/106613332/c70dbadf-777e-4796-9aa9-c2187b57e382)
---

If you use this code, please cite our paper:

```
@article{Yue24robust,
title = {Robust pixel-wise illuminant estimation algorithm for images with a low bit-depth},
author = {Shuwei Yue and Minchen Wei},
journal = {Opt. Express},
number = {15},
pages = {26708--26718},
volume = {32},
month = {Jul},
year = {2024},
publisher = {Optica Publishing Group}

```

## Pre-requites
- Download the LSMI dataset(The origin dataset is large which may taken you more than one day to process, I released the **tiff** format in this link)
- Download the pre-trained models and put them into the 'pretrained_models' fold, link 

## Code
The Net architecture is the same as *[LIMIU](https://github.com/DY112/LSMI-dataset) 

Our key contribution is the *post_processing.py* as the physical-constrained post-processing, detailed in the paper.

### Train
Check the path in 'setting.py' and run 'train.py'

### Test
Check the path in 'test.py' and run 'test.py'. Default is using post-processing in **USING_POST_PROCESSING=True**, you can change it to **False** for comparison, and you will see an amazing improvement!


---

Please do not hesitate to contact me if you have any questions, enjoy!

# DRRN-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation of [Image Super-Resolution via Deep Recursive Residual Network](https://openaccess.thecvf.com/content_cvpr_2017/papers/Tai_Image_Super-Resolution_via_CVPR_2017_paper.pdf).

### Table of contents

- [DRRN-PyTorch](#drrn-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [About Image Super-Resolution via Deep Recursive Residual Network](#about-image-super-resolution-via-deep-recursive-residual-network)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [Test](#test)
    - [Train](#train)
    - [Result](#result)
    - [Credit](#credit)
        - [Image Super-Resolution via Deep Recursive Residual Network](#image-super-resolution-via-deep-recursive-residual-network)

## About Image Super-Resolution via Deep Recursive Residual Network

If you're new to DRRN, here's an abstract straight from the paper:

Recently, Convolutional Neural Network (CNN) based models have achieved great success in Single Image SuperResolution (SISR). Owing to the strength of
deep networks, these CNN models learn an effective nonlinear mapping from the low-resolution input image to the high-resolution target image, at the
cost of requiring enormous parameters. This paper proposes a very deep CNN model (up to 52 convolutional layers) named Deep Recursive Residual Network
(DRRN) that strives for deep yet concise networks. Specifically, residual learning is adopted, both in global and local manners, to mitigate the
difficulty of training very deep networks; recursive learning is used to control the model parameters while increasing the depth. Extensive benchmark
evaluation shows that DRRN significantly outperforms state of the art in SISR, while utilizing far fewer parameters. Code is available
at https://github.com/tyshiwo/DRRN CVPR17.

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

## Test

Modify the contents of the file as follows.

- line 29: `upscale_factor` change to the magnification you need to enlarge.
- line 31: `num_residual_unit` change to Residual Neural Network Depth.
- line 33: `mode` change Set to valid mode.
- line 76: `model_path` change weight address after training.

## Train

Modify the contents of the file as follows.

- line 29: `upscale_factor` change to the magnification you need to enlarge.
- line 31: `num_residual_unit` change to Residual Neural Network Depth.
- line 33: `mode` change Set to train mode.

If you want to load weights that you've trained before, modify the contents of the file as follows.

- line 49: `start_epoch` change number of training iterations in the previous round.
- line 50: `resume` change weight address that needs to be loaded.

## Result

Source of original paper results: https://openaccess.thecvf.com/content_cvpr_2017/papers/Tai_Image_Super-Resolution_via_CVPR_2017_paper.pdf

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale | (DRRN_B1U9) PSNR | (DRRN_B1U25) PSNR |
|:-------:|:-----:|:----------------:|:-----------------:|
|  Set5   |   2   | 37.66(**37.56**) | 37.74(**37.50**)  |
|  Set5   |   3   | 33.93(**33.75**) | 34.03(**33.75**)  |
|  Set5   |   4   | 31.58(**31.34**) | 31.68(**31.38**)  |

Low Resolution / Super Resolution / High Resolution
<span align="center"><img src="assets/result.png"/></span>

### Credit

#### Image Super-Resolution via Deep Recursive Residual Network

_Ying Tai, Jian Yang1, Xiaoming Liu_ <br>

**Abstract** <br>
Recently, Convolutional Neural Network (CNN) based models have achieved great success in Single Image SuperResolution (SISR). Owing to the strength of
deep networks, these CNN models learn an effective nonlinear mapping from the low-resolution input image to the high-resolution target image, at the
cost of requiring enormous parameters. This paper proposes a very deep CNN model (up to 52 convolutional layers) named Deep Recursive Residual Network
(DRRN) that strives for deep yet concise networks. Specifically, residual learning is adopted, both in global and local manners, to mitigate the
difficulty of training very deep networks; recursive learning is used to control the model parameters while increasing the depth. Extensive benchmark
evaluation shows that DRRN significantly outperforms state of the art in SISR, while utilizing far fewer parameters. Code is available
at https://github.com/tyshiwo/DRRN CVPR17.

[[Paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Tai_Image_Super-Resolution_via_CVPR_2017_paper.pdf) [[Author's implements(Caffe)]](https://github.com/tyshiwo/DRRN_CVPR17)

```
@inproceedings{Tai-DRRN-2017,
  title={Image Super-Resolution via Deep Recursive Residual Network},
  author={Tai, Ying and Yang, Jian and Liu, Xiaoming },
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
```

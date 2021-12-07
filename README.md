# DRRN-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation
of [Image Super-Resolution via Deep Recursive Residual Network](https://openaccess.thecvf.com/content_cvpr_2017/papers/Tai_Image_Super-Resolution_via_CVPR_2017_paper.pdf)
.

### Table of contents

- [DRRN-PyTorch](#drrn-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [About Image Super-Resolution via Deep Recursive Residual Network](#about-image-super-resolution-via-deep-recursive-residual-network)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
        - [Download train dataset](#download-train-dataset)
        - [Download valid dataset](#download-valid-dataset)
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

- [Google Driver](https://drive.google.com/drive/folders/1yLlwp-W-VTqSPbR7QispSfosLdKEz6Wg?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1Hk7iEpsvuw-DXHEKTj9RMw) access:`llot`

## Download datasets

### Download train dataset

#### TB291

- Image format
    - [Google Driver](https://drive.google.com/drive/folders/13wiE6YqIhyix0RFxpFONJ7Zz_00CttdX?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1mhbFj0Nvwthmgx07Gas5BQ) access: `llot`

- LMDB format (train)
    - [Google Driver](https://drive.google.com/drive/folders/1BPqN08QHk_xFnMJWMS8grfh_vesVs8Jf?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1eqeORnKcTmGatx2kAG92-A) access: `llot`

- LMDB format (valid)
    - [Google Driver](https://drive.google.com/drive/folders/1bYqqKk6NJ9wUfxTH2t_LbdMTB04OUicc?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1W34MeEtLY0m-bOrnaveVmw) access: `llot`

### Download valid dataset

#### Set5

- Image format
    - [Google Driver](https://drive.google.com/file/d/1GtQuoEN78q3AIP8vkh-17X90thYp_FfU/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1dlPcpwRPUBOnxlfW5--S5g) access:`llot`

#### Set14

- Image format
    - [Google Driver](https://drive.google.com/file/d/1CzwwAtLSW9sog3acXj8s7Hg3S7kr2HiZ/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1KBS38UAjM7bJ_e6a54eHaA) access:`llot`

#### BSD200

- Image format
    - [Google Driver](https://drive.google.com/file/d/1cdMYTPr77RdOgyAvJPMQqaJHWrD5ma5n/view?usp=sharing)
    - [Baidu Driver](https://pan.baidu.com/s/1xahPw4dNNc3XspMMOuw1Bw) access:`llot`

## Test

Modify the contents of the file as follows.

- line 24: `upscale_factor` change to the magnification you need to enlarge.
- line 25: `mode` change Set to valid mode.
- line 85: `model_path` change weight address after training.

## Train

Modify the contents of the file as follows.

- line 24: `upscale_factor` change to the magnification you need to enlarge.
- line 25: `mode` change Set to train mode.

If you want to load weights that you've trained before, modify the contents of the file as follows.

- line 56: `resume` change to `True`.
- line 57: `strict` Transfer learning is set to `False`, incremental learning is set to `True`.
- line 58: `start_epoch` change number of training iterations in the previous round.
- line 59: `resume_weight` the weight address that needs to be loaded.

## Result

Source of original paper results: https://openaccess.thecvf.com/content_cvpr_2017/papers/Tai_Image_Super-Resolution_via_CVPR_2017_paper.pdf

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale | (DRRN_B1U9) PSNR | (DRRN_B1U25) PSNR |
|:-------:|:-----:|:----------------:|:-----------------:|
|  Set5   |   2   | 37.66(**37.56**) | 37.74(**37.50**)  |
|  Set5   |   3   | 33.93(**33.75**) | 34.03(**33.75**)  |
|  Set5   |   4   | 31.58(**31.34**) | 31.68(**31.34**)  |

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

[[Paper]](https://arxiv.org/pdf/1511.04587)

```
@inproceedings{Tai-DRRN-2017,
  title={Image Super-Resolution via Deep Recursive Residual Network},
  author={Tai, Ying and Yang, Jian and Liu, Xiaoming },
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
```

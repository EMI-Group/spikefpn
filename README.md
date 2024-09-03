# SpikeFPN: Automotive Object Detection via Learning Sparse Events by Spiking Neurons

<p align="center">
  <a href="https://arxiv.org/abs/2307.12900">
    <img src="https://img.shields.io/badge/paper-arxiv-red?style=for-the-badge" alt="SpikeFPN Paper on arXiv">
  </a>
</p>

This work explores the membrane potential dynamics of spiking neural networks (SNNs) and their ability to modulate sparse events. We introduce an innovative spike-triggered adaptive threshold mechanism designed for stable training. Building on these insights, we present a specialized spiking feature pyramid network (SpikeFPN) optimized for automotive event-based object detection. Comprehensive evaluations demonstrate that SpikeFPN surpasses both traditional SNN and advanced artificial neural network (ANN) models.

## Environment Configuration

In a configuration utilizing `Ubuntu 22.04`, `CUDA 12.4`, and `PyTorch 2.3.1`:

```shell
apt-get update # If necessary
apt-get install ffmpeg libsm6 libxext6
pip install -r requirements.txt
```

## Experiment on GEN1 Automotive Detection (GAD) Dataset

### Data Preprocessing
```shell
python ./preprocess/gad_framing.py
```

### Training and Testing
```shell
python ./train_gad.py
python ./test_gad.py
```



## Experiment on N-CARS Dataset

### Data Preprocessing
```shell
python ./preprocess/ncars_framing.py
```

### Data Division
|                    | Class: background | Class: cars  |
| :----------------- | :---------------: | :----------: |
| **For Training**   |     0 ~ 4210      |   0 ~ 4395   |
| **For Validating** |    4211 ~ 5706    | 4396 ~ 5983  |
| **For Testing**    |   5707 ~ 11692    | 5984 ~ 12335 |

### Training and Testing
```shell
python ./train_ncars.py
python ./test_ncars.py
```

## Citing SpikeFPN

```latex
@ARTICLE{spikefpn,
    author={Zhang, Hu and Li, Yanchen and Leng, Luziwei and Che, Kaiwei and Liu, Qian and Guo, Qinghai and Liao, Jianxing and Cheng, Ran},
    journal={IEEE Transactions on Cognitive and Developmental Systems}, 
    title={Automotive Object Detection via Learning Sparse Events by Spiking Neurons}, 
    year={2024},
    pages={1-15},
    doi={10.1109/TCDS.2024.3410371}
}
```


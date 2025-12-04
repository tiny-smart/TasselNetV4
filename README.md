# TasselNetV4: A vision foundation model for cross-scene, cross-scale, and cross-species plant counting

Official implementation of [TasselNetV4](https://arxiv.org/abs/2509.20857)

Accepted by [International Society for Photogrammetry and Remote Sensing](https://www.sciencedirect.com/science/article/pii/S0924271625004575)(IF=12.2). Many thanks to all authors and reviewers`:smile:`

Xiaonan Hu, Xuebing Li, Jinyu Xu, Abdulkadir Duran Adan, Xuhui Zhu, Yanan Li, Wei Guo, Shouyang Liu, Wenzhong Liu, Hao Lu


## Highlights
- **Plant agnostic counting:** a new plant-orientated task customizing Class-agnostic counting into the plant domain and highlighting zero-shot generalization across taxomomic plant species;
- **PAC-105 and PAC-Somalia:** two challenging PAC datasets for training and evaluating daily and out-of-distribution plant species;
- **TasselNetV4:** an extended version of the TasselNet plant counting models.

![motivation](/assets/motivation.png "plant-specific counting to plant-agnostic counting")

### Comparison with the state-of-the-art CAC approaches on the PAC-105 dataset. Best performance is in boldface.


| Method                     | Venue & Year     | Shot        | MAE↓  | RMSE↓  | WCA↑ | $R^2$ ↑  |
|----------------------------|-----------------|-------|-------|--------|------|------|
| FamNet([checkpoint](https://pan.baidu.com/s/1QVJcZA2CELPf9aRS5QXDPQ?pwd=bg5y))    | CVPR'21         | 3   | 31.70 | 62.58  | 0.49 | 0.56 |
| BMNet+([checkpoint](https://pan.baidu.com/s/1cKjICAi4WDShRlheK9b3wA?pwd=qjp9))    | CVPR'22         | 3   | 27.03 | 60.18  | 0.56 | 0.61 |
| SPDCNet([checkpoint](https://pan.baidu.com/s/1YM9caohZKS5ERk8XBFL5Pw?pwd=xshm))   | BMVC'22         | 3   | 25.21 | 49.98  | 0.58 | 0.92 | 
| SAFECount([checkpoint](https://pan.baidu.com/s/1Y3KormYsO6hAFEYD3zRFjQ?pwd=swv5)) | WACV'23         | 3   | 25.59 | 52.09  | 0.58 | 0.91 | 
| CountTR([checkpoint](https://pan.baidu.com/s/1ASJCBc3QK8TR-uItlZcs8A?pwd=94at))   | BMVC'22         | 3   | 25.25 | 49.31  | 0.63 | 0.92 | 
| T-Rex2    | ECCV'24         | 3   | 26.04 | 49.31  | 0.58 | 0.92 | 
| CACViT([checkpoint](https://pan.baidu.com/s/1qpVekxoPaMo30dV1wfXhIA?pwd=s26c))    | AAAI'24         | 3   | 19.51 | 29.59  | 0.68 | 0.89 | 
| **TasselNetV4 (Ours)**      | ISPRS'25       | 3   | **16.04** | **28.03** | **0.74** | **0.92** |
| FamNet    | CVPR'21         | 1   | 35.91±0.966 | 71.78±1.188 | 0.42±0.014 | 0.45±0.024 |
| BMNet     | CVPR'22         | 1   | 28.78±0.324 | 62.12±0.437 | 0.15±0.001 | 0.59±0.008 |
| CountTR   | BMVC'22         | 1   | 28.46±0.226 | 49.84±0.646 | 0.70±0.037 | 0.73±0.006 |
| CACViT    | AAAI'24         | 1   | 21.80±0.429 | 38.40±1.526 | 0.64±0.005 | 0.84±0.013 |
| **TasselNetV4 (Ours)**      | ISPRS'25       | 1   | **18.04±0.339** | **32.04±1.213** | **0.71±0.005** | **0.90±0.009** |



### Comparison with the state-of-the-art CAC approaches on the PAC-Somalia dataset. Best performance is in boldface.

| Method                     | Venue & Year     | Shot     | MAE↓  | RMSE↓  | WCA↑ | $R^2$ ↑  |
|----------------------------|-----------------|-------|--------|------|------|------|
| CountTR                   | BMVC'22         | 3     | 12.71 | 23.87  | 0.38 | 0.57 |
| CACViT                    | AAAI'24      | 3     | 14.00 | 17.00  | 0.55 | 0.78 |
| **TasselNetV4 (Ours)**    | This Paper      | 3     | **8.88** | **13.11** | **0.72** | **0.87** |
| CountTR                   | BMVC'22         | 1     | 12.79±0.076 | 24.20±0.292 | 0.37±0.005 | 0.55±0.012 |
| CACViT                    | AAAI'24         | 1     | 14.74±0.138 | 18.23±0.446 | 0.53±0.005 | 0.75±0.012 |
| **TasselNetV4 (Ours)**    | ISPRS'25     | 1     | **10.98±0.065** | **16.73±0.150** | **0.65±0.000** | **0.80±0.005** |


## Visualization
![visualization](/assets/visualization.png "Visualization of baselines and our method")

## Installation
To setup all the required dependencies for training and evaluation, please follow the instructions below:

```bash
conda env create -f environment.yaml
conda activate TN4
```


## Prepare Dataset

**PAC-105&PAC-Somalia**

Download training dataset PAC-105 from [Baiduyun (2.8G)](https://pan.baidu.com/s/1gB78ekTl-cHbkoIyZMtiYA?pwd=hgxp) | [Google Drive (2.8G)](https://drive.google.com/file/d/1IB6yPXEvXQN3AbAYPjprHb0xsGP0BeHq/view?usp=drive_link) and test dataset PAC-Somalia from: [Baiduyun (208M)](https://pan.baidu.com/s/1UH0rihsMe06_5J8AtALTeg?pwd=jssy) | [Google Drive (208M)](https://drive.google.com/file/d/1-haH0eZdpcOK9IMGkw0UafWq_YioHIyn/view?usp=drive_link).The dataset structure should look like this:
````
/dataset
├──── aska
    ├──── aska_fruit
        ├──── images
            ├──── image1.png
            └──── ...
        └──── labels
            └──── aska.csv
├──── boocbooc
├──── ......
├──── yicib
└──── dataset.csv
````

## Inference
First modify *Dataset.data_path* and *Resume.resume_path* in `cfg_test/local_32_64_128_loose.yml`;

Download our model from [Baiduyun (1.05G)](https://pan.baidu.com/s/1B71m2TDmClENOevlTUxC1w?pwd=qdgr) | [Google Drive (1.05G)](https://drive.google.com/file/d/1vrHBv8c1PFJjOX_IXPe0K0oXYNkY7a__/view?usp=drive_link);

Run the following command to reproduce our results of TasselNetV4 on the PAC-105 / PAC-Somalia:

`python test_checkpoint.py`
    
- Results will be saved in the path `./visual`.
  
<!-- ## Training
First modify path in `cfg_train/local_32_64_128_loose.yml`;

Run the following command to train your model

`python train_val.py`
     -->

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE)

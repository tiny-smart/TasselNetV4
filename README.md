# TasselNetv4


Official implementation of [TasselNetV4: Towards plant-agnostic counting with a plain vision transformer and box-aware local counters]
<!-- <p align="center">
  <img src="plant_counting.png" width="825"/>
</p> --> 

## Highlights
- **Plant agnostic counting:** 
- **PAC-105:** 
- **PAC-Somalia:**
### Comparison with the state-of-the-art CAC approaches on the PAC-105 dataset with $3$-shots. Best performance is in boldface.

| Method                     | Venue & Year     | MAE↓  | RMSE↓  | WCA↑ | $R^2$ ↑  | FPST  |
|----------------------------|-----------------|-------|--------|------|------|-------|
| FamNet (Ranjan et al., 2021)  | CVPR'21         | 31.70 | 62.58  | 0.49 | 0.56 | 89.65  |
| BMNet (Shi et al., 2022)      | CVPR'22         | 27.03 | 60.18  | 0.56 | 0.61 | 43.08  |
| SPDCNet (Lin et al., 2022)    | BMVC'22         | 25.21 | 49.98  | 0.58 | 0.92 | 155.84 |
| SAFECount (You et al., 2023)  | WACV'23         | 25.59 | 52.09  | 0.58 | 0.91 | 118.55 |
| CountTR (Liu et al., 2022)    | BMVC'22         | 25.25 | 49.31  | 0.63 | 0.92 | 112.46 |
| T-Rex2 (Jiang et al., 2025)   | ECCV'24         | 26.04 | 49.31  | 0.58 | 0.92 | \      |
| CACViT (Wang et al., 2024b)      | AAAI'24         | 19.51 | 29.59  | 0.68 | 0.89 | 89.65  |
| **TasselNetV4 (Ours)**        | This Paper      | **16.04** | **28.03** | **0.74** | **0.92** | **121.62** |


### Comparison with the state-of-the-art CAC approaches on the PAC-105 dataset with 1-shot. Best performances in boldface.

| Method                     | Venue & Year     | MAE↓  | RMSE↓  | WCA↑ | $R^2$ ↑  |
|----------------------------|-----------------|-------|--------|------|------|
| FamNet (Ranjan et al., 2021)  | CVPR'21         | 35.89±0.97 | 71.78±1.19 | 0.42±0.014 | 0.450±0.024 |
| BMNet (Shi et al., 2022)      | CVPR'22         | 28.78±0.32 | 61.91±0.48 | 0.51±0.012 | 0.578±0.008 |
| CountTR (Liu et al., 2022)    | BMVC'22         | 28.46±0.23 | 49.84±0.65 | 0.70±0.037 | 0.73±0.007 |
| CACViT  (Wang et al., 2024b)  | AAAI'24         | 22.66±0.67 | 37.43±4.23 | 0.63±0.009 | 0.85±0.034 |
| **TasselNetV4 (Ours)**        | This Paper      | **18.04±0.34** | **32.04±0.12** | **0.71±0.005** | **0.90±0.009** |

### Comparison with the state-of-the-art CAC approaches on the PAC-Somalia dataset with $3$-shots. Best performance is in boldface.

| Method                     | Venue & Year     | MAE↓  | RMSE↓  | WCA↑ | $R^2$ ↑  |
|----------------------------|-----------------|-------|--------|------|------|
| CountTR (Liu et al., 2022)    | BMVC'22         | 12.71 | 23.87  | 0.38 | 0.57 |
| CACViT (Wang et al., 2024b)      | AAAI'24         | 14.00 | 17.00  | 0.55 | 0.78 |
| **TasselNetV4 (Ours)**        | This Paper      | **8.88** | **13.11** | **0.72** | **0.87** |

### Comparison with the state-of-the-art CAC approaches on the PAC-Somalia dataset with 1-shot. Best performances in boldface.

| Method                     | Venue & Year     | MAE↓  | RMSE↓  | WCA↑ | $R^2$ ↑  |
|----------------------------|-----------------|-------|--------|------|------|
| CountTR (Liu et al., 2022)    | BMVC'22         | 12.79±0.076 | 24.20±0.29 | 0.37±0.0047 | 0.55±0.012 |
| CACViT  (Wang et al., 2024b)  | AAAI'24         | 14.74±0.13 | 18.23±0.45 | 0.53±0.0047 | 0.75±0.012 |
| **TasselNetV4 (Ours)**        | This Paper      | **10.98±0.065** | **16.73±0.15** | **0.65±0.00** | **0.80±0.0047** |

## Visualization
![visualization](/assets/visualization.png "Visualization of baselines and our method")

## Installation
See details in `environment.yaml`.
`conda env create -f environment.yaml`


## Prepare Dataset
**PAC-105**
Download PAC-105 dataset from: [Google Drive (2.5 GB)](https://drive.google.com/open?id=1XHcTqRWf-xD-WuBeJ0C9KfIN8ye6cnSs).The dataset structure should look like this:
````
/PAC_105
├──── almond
    ├──── almond_fruit
        ├──── images
            ├──── image1.png
            └──── ...
        └──── labels
            └──── almond.csv
├──── apple
├──── ......
├──── wheat
└──── dataset.csv
````

**Supplymentary dataset**
Download PAC-105 supplymentary subset from: [Google Drive (2.5 GB)](https://drive.google.com/open?id=1XHcTqRWf-xD-WuBeJ0C9KfIN8ye6cnSs).The dataset structure should look like this:
````
/PAC_105
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
First modify Dataset[data_path]/Resume[resume_path] in `cfg_test/local_32_64_128_loose.yml`;
Run the following command to reproduce our results of TasselNetV4 on the PAC-105 dataset/PAC-105 Supplymentary dataset:

`python test_checkpoint.py`
    
- Results are saved in the path `./visual`.
  
## Training
First modify path in `cfg_train/local_32_64_128_loose.yml`;
Download pretrain model from:
Run the following command to 
`python train_val.py`
    

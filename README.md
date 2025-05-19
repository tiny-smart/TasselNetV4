# TasselNetv4


Official implementation of [TasselNetV4: Towards plant-agnostic counting with a plain vision transformer and box-aware local counters]

## Highlights
- **Plant agnostic counting:** a new plant-orientated task customizing Class-agnostic counting into the plant domain and highlighting zero-shot generalization across taxomomic plant species;
- **PAC-105 and PAC-Somalia:** two challenging PAC datasets for training and evaluating daily and out-of-distribution plant species;
- **TasselNetV4:** an extended version of the TasselNet plant counting models.
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
See details in 
`environment.yaml`.

Or simply run the following code:

`conda env create -f environment.yaml`


## Prepare Dataset
**PAC-105**
Download PAC-105 dataset from: [Baiduyun (2.80G)](https://pan.baidu.com/s/1Zl3LYwGZOjUaF9K_DjgH2A?pwd=7sgv) | [Google Drive (2.80G)](https://drive.google.com/file/d/1Y5ca8Eb3hOhimFfduiYN_rcJeL1CAVGU/view?usp=drive_link).The dataset structure should look like this:
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

**PAC-Somalia**
Download PAC-Somalia dataset from: [Baiduyun (208M)](https://pan.baidu.com/s/1UH0rihsMe06_5J8AtALTeg?pwd=jssy) | [Google Drive (208M)](https://drive.google.com/file/d/1-haH0eZdpcOK9IMGkw0UafWq_YioHIyn/view?usp=drive_link).The dataset structure should look like this:
````
/PAC_Somalia
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

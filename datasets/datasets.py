import os
if __name__ != '__main__':
    from datasets.plant import PlantDataset,Train_dataset,Val_Test_dataset,resizeTrainImage
else:
    import sys
    sys.path.append(os.getcwd())
    from datasets.plant import PlantDataset,Train_dataset,Val_Test_dataset,resizeTrainImage
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

from torch.utils.data import ConcatDataset

IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]
MAX_HW = 384
def build_dataloader(flag,**cfg):
    data = PlantDataset(data_path=cfg['data_path'])
    TransformTrain = transforms.Compose([resizeTrainImage(dataset=data,MAX_HW=MAX_HW, local=cfg['local'])])
    if flag == 'train':
        dataset = Train_dataset(transform_train=TransformTrain,dataset=data)
    else :
        if 'validation' in cfg.keys():
            temp = []
            for split in cfg['validation']:
                temp.append(Val_Test_dataset(dataset=data,mode=split))
            dataset = ConcatDataset(temp)
        else:
            dataset = Val_Test_dataset(dataset=data,mode=flag)
    sampler = torch.utils.data.RandomSampler(dataset)
    print(f"build dataset_{flag} from {cfg['data_path']}done, nums:{len(dataset)}")
    if flag == 'train':
        return DataLoader(
            dataset=dataset,
            batch_size=cfg['batch_size'],
            num_workers=cfg['num_workers'],
            sampler=sampler,
            pin_memory=cfg['pin_memory'],
            drop_last=False
        )
    else:
        return DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            pin_memory=cfg['pin_memory'],
            drop_last=False,
        )
    
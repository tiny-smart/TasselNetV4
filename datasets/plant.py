from torch.utils.data import Dataset
import pandas as pd
import os
import random
from PIL import Image
import math
import numpy as np
import torch
from torchvision import transforms
import cv2
import torch.nn as nn
import torchvision.transforms.functional as TF
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from PIL import Image



MAX_HW = 384
MAX_W = 384
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]
# val_category = [
#     'apricot_flower',
#     # 'jackfruit',
#     'lettuce_plant',
#     'sorghum_head',
#     'tulip_flower',
#     'Arvalis_4',
#     'Arvalis_9',
#     'baima_wheat_seedling',
#     'ETHZ_1',
#     'NAU_2',
#     'NMBU_1',
#     'NMBU_2',
#     'Terraref_1',
#     'Ukyoto_1',
#     'ULiege-GxABT_1',
#     'Usask_1',
#     'Utokyo_2'
# ]
class PlantDataset(object):
    def __init__(self, data_path) :
        data_split_file = data_path + 'dataset.csv'
        annotations = {}
        class_dict={}
        data_split={'train':[], 'val':[], 'test':[]}
        data_split_csv = pd.read_csv(data_split_file)
        for i in range(len(data_split_csv)):
            key_split = data_split_csv['split'][i]
            labelpath = data_path+data_split_csv['variety'][i]+'/'+data_split_csv['foldname'][i]+'/labels/'
            # just for 
            # if not data_split_csv['foldname'][i] in val_category:
            #     continue
            imgpath = data_path+data_split_csv['variety'][i]+'/'+data_split_csv['foldname'][i]+'/images/'
            for img in os.listdir(imgpath):
                value_split = data_split_csv['variety'][i]+'/'+data_split_csv['foldname'][i]+'/images/'+img
                data_split[key_split].append(value_split)
                class_dict[value_split]=data_split_csv['variety'][i]+'/'+data_split_csv['foldname'][i]

            for file in os.listdir(labelpath):
                data_fold_csv = pd.read_csv(labelpath+file)
                for k in range(len(data_fold_csv)):
                    key_fold = data_split_csv['variety'][i]+'/'+data_split_csv['foldname'][i]+'/images/'+data_fold_csv['filename'][k]
                    if key_fold not in annotations:
                        annotations[key_fold] = {'box_examples_coordinates':[], 'points':[]}
                    if data_fold_csv['region_shape_attributes'][k].find("rect") != -1:
                        placex1 = data_fold_csv['region_shape_attributes'][k].find(":", 9)
                        placex2 = data_fold_csv['region_shape_attributes'][k].find(",", placex1)
                        placey1 = data_fold_csv['region_shape_attributes'][k].find(":", placex2)
                        placey2 = data_fold_csv['region_shape_attributes'][k].find(",", placey1)
                        placew1 = data_fold_csv['region_shape_attributes'][k].find(":", placey2)
                        placew2 = data_fold_csv['region_shape_attributes'][k].find(",", placew1)
                        placeh1 = data_fold_csv['region_shape_attributes'][k].find(":", placew2)
                        placeh2 = data_fold_csv['region_shape_attributes'][k].find("}", placeh1)
                        x1 = round(float(data_fold_csv['region_shape_attributes'][k][placex1 + 1:placex2]))
                        y1 = round(float(data_fold_csv['region_shape_attributes'][k][placey1 + 1:placey2]))
                        w = round(float(data_fold_csv['region_shape_attributes'][k][placew1 + 1:placew2]))
                        h = round(float(data_fold_csv['region_shape_attributes'][k][placeh1 + 1:placeh2]))
                        x2 = x1 + w
                        y2 = y1 + h
                        annotations[key_fold]['box_examples_coordinates'].append([[x1,y1], [x1, y2], [x2, y2], [x2, y1]])
                    elif data_fold_csv['region_shape_attributes'][k].find("point") != -1:
                        placex1 = data_fold_csv['region_shape_attributes'][k].find(":", 9)
                        placex2 = data_fold_csv['region_shape_attributes'][k].find(",", placex1)
                        placey1 = data_fold_csv['region_shape_attributes'][k].find(":", placex2)
                        placey2 = data_fold_csv['region_shape_attributes'][k].find("}", placey1)
                        x1 = round(float(data_fold_csv['region_shape_attributes'][k][placex1 + 1:placex2]))
                        y1 = round(float(data_fold_csv['region_shape_attributes'][k][placey1 + 1:placey2]))
                        annotations[key_fold]['points'].append([x1, y1])

        self.annotations = annotations
        self.data_split = data_split
        self.data_path = data_path
        self.class_dict = class_dict

        # self.data_split['test'].update(self.data_split['val'])
        # self.data_split['test'].update(self.data_split['train'])

def getsize(size:torch.tensor):
    # import pdb
    # pdb.set_trace()
    sizemax = size.max()
    
    # sy = torch.mean(size, dim=1)
    return math.sqrt(sizemax.item() * sizemax.item() * 384 * 384)


class PlantDatasetSpecificClass(object):
    def __init__(self, data_path) :
        data_split_file = data_path + 'dataset.csv'
        annotations = {}
        class_dict={}
        data_split={'train':{}, 'val':{}, 'test':{}}
        data_split_csv = pd.read_csv(data_split_file)
        for i in range(len(data_split_csv)):
            key_split = data_split_csv['split'][i]
            labelpath = data_path+data_split_csv['variety'][i]+'/'+data_split_csv['foldname'][i]+'/labels/'
            # just for 
            # if not data_split_csv['foldname'][i] in val_category:
            #     continue
            imgpath = data_path+data_split_csv['variety'][i]+'/'+data_split_csv['foldname'][i]+'/images/'
            for img in os.listdir(imgpath):
                value_split = data_split_csv['variety'][i]+'/'+data_split_csv['foldname'][i]+'/images/'+img
                if data_split_csv['foldname'][i] in data_split[key_split].keys():
                    data_split[key_split][data_split_csv['foldname'][i]].append(value_split)
                else:
                    data_split[key_split].update({data_split_csv['foldname'][i]: [value_split]})
                # data_split[key_split].append(value_split)
                class_dict[value_split]=data_split_csv['variety'][i]+'/'+data_split_csv['foldname'][i]

            for file in os.listdir(labelpath):
                data_fold_csv = pd.read_csv(labelpath+file)
                for k in range(len(data_fold_csv)):
                    key_fold = data_split_csv['variety'][i]+'/'+data_split_csv['foldname'][i]+'/images/'+data_fold_csv['filename'][k]
                    if key_fold not in annotations:
                        annotations[key_fold] = {'box_examples_coordinates':[], 'points':[]}
                    if data_fold_csv['region_shape_attributes'][k].find("rect") != -1:
                        placex1 = data_fold_csv['region_shape_attributes'][k].find(":", 9)
                        placex2 = data_fold_csv['region_shape_attributes'][k].find(",", placex1)
                        placey1 = data_fold_csv['region_shape_attributes'][k].find(":", placex2)
                        placey2 = data_fold_csv['region_shape_attributes'][k].find(",", placey1)
                        placew1 = data_fold_csv['region_shape_attributes'][k].find(":", placey2)
                        placew2 = data_fold_csv['region_shape_attributes'][k].find(",", placew1)
                        placeh1 = data_fold_csv['region_shape_attributes'][k].find(":", placew2)
                        placeh2 = data_fold_csv['region_shape_attributes'][k].find("}", placeh1)
                        x1 = round(float(data_fold_csv['region_shape_attributes'][k][placex1 + 1:placex2]))
                        y1 = round(float(data_fold_csv['region_shape_attributes'][k][placey1 + 1:placey2]))
                        w = round(float(data_fold_csv['region_shape_attributes'][k][placew1 + 1:placew2]))
                        h = round(float(data_fold_csv['region_shape_attributes'][k][placeh1 + 1:placeh2]))
                        x2 = x1 + w
                        y2 = y1 + h
                        annotations[key_fold]['box_examples_coordinates'].append([[x1,y1], [x1, y2], [x2, y2], [x2, y1]])
                    elif data_fold_csv['region_shape_attributes'][k].find("point") != -1:
                        placex1 = data_fold_csv['region_shape_attributes'][k].find(":", 9)
                        placex2 = data_fold_csv['region_shape_attributes'][k].find(",", placex1)
                        placey1 = data_fold_csv['region_shape_attributes'][k].find(":", placex2)
                        placey2 = data_fold_csv['region_shape_attributes'][k].find("}", placey1)
                        x1 = round(float(data_fold_csv['region_shape_attributes'][k][placex1 + 1:placex2]))
                        y1 = round(float(data_fold_csv['region_shape_attributes'][k][placey1 + 1:placey2]))
                        annotations[key_fold]['points'].append([x1, y1])

        self.annotations = annotations
        self.data_split = data_split
        self.data_path = data_path
        self.class_dict = class_dict
        self.category = list(data_split_csv['variety'])

        # self.data_split['test'].update(self.data_split['val'])
        # self.data_split['test'].update(self.data_split['train'])

class Train_dataset(Dataset):
    def __init__(self,transform_train,dataset:PlantDataset):
        self.annotations = dataset.annotations
        self.img = dataset.data_split['train']
        self.img_dir = dataset.data_path
        self.Transform = transform_train

    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
        im_id = self.img[idx]
        anno = self.annotations[im_id]
        bboxes = anno['box_examples_coordinates']

        rects = list()
        examplar_size_list = []

        image = Image.open('{}{}'.format(self.img_dir, im_id))
        w,h = image.size
        
        t_flag = 0
        image.load()
        if w < h:
            image = image.transpose(Image.TRANSPOSE)
            t_flag = 1
        w, h = image.size
        m_flag = 0
        resize_ratio = 1
        # print(image.size)
        newh = 384
        resize_ratio = newh/h
        neww = 16 * int(math.ceil(w * resize_ratio) / 16)
        image=image.resize((neww, newh), resample=Image.BICUBIC)
        if t_flag == 0:
            for bbox in bboxes:
                x1 = round(bbox[0][0] * resize_ratio)
                y1 = round(bbox[0][1] * resize_ratio)
                x2 = round(bbox[2][0] * resize_ratio)
                y2 = round(bbox[2][1] * resize_ratio)
                rects.append([y1, x1, y2, x2])
                examplar_size_list.append((y2-y1, x2-x1))
        else:
            for bbox in bboxes:
                y1 = round(bbox[0][0] * resize_ratio)
                x1 = round(bbox[0][1] * resize_ratio)
                y2 = round(bbox[2][0] * resize_ratio)
                x2 = round(bbox[2][1] * resize_ratio)
                rects.append([y1, x1, y2, x2])
                examplar_size_list.append((y2-y1, x2-x1))
        dots = np.array(anno['points'],dtype=np.float32)
        gt_map = np.zeros((newh,neww),dtype='float32')

        if t_flag == 1:
            for i in range(dots.shape[0]):
                ptx = float(dots[i][1])
                pty = float(dots[i][0])
                dots[i][0] = ptx * resize_ratio
                dots[i][1] = pty * resize_ratio
                gt_map[min(round(float(dots[i][1])), newh - 1)][
                    min(round(float(dots[i][0])), neww - 1)] = 1
        else:
            for i in range(dots.shape[0]):
                dots[i][0] = float(dots[i][0]) * resize_ratio
                dots[i][1] = float(dots[i][1]) * resize_ratio
                gt_map[min(round(float(dots[i][1])), newh - 1)][
                    min(round(float(dots[i][0])), neww - 1)] = 1
        gt_map = torch.from_numpy(gt_map)
        density = np.array(gt_map)

        sample = {'image':image,'lines_boxes':rects,'gt_density':density,
                  'dots':dots, 'id':im_id, 'm_flag': m_flag, 'idx': idx}
        # print(f"idx: {idx}, image size: {image}, examplar size: {examplar_size_list}")
        sample = self.Transform(sample)
        sample['examplar_size'] = examplar_size_list
        return sample
    
class Val_Test_dataset(Dataset):
    def __init__(self,dataset:PlantDataset,mode:str,local=False):
        self.annotations = dataset.annotations
        self.img = dataset.data_split[mode]
        self.img_dir = dataset.data_path
        # for img in self.img_dir:
        #     if not img in val_category:


    def __len__(self):
        return len(self.img)
    

    def __getitem__(self, idx):
        im_id = self.img[idx]
        anno = self.annotations[im_id]
        bboxes = anno['box_examples_coordinates']

        dots = np.array(anno['points'])
        try:
            image = Image.open('{}{}'.format(self.img_dir, im_id))
            if image.mode != 'RGB':
                image = image.convert('RGB')
                print(im_id)

            w, h = image.size
            image.load()
        except:
            print(im_id)
        t_flag = 0
        if w < h:
            image = image.transpose(Image.TRANSPOSE)
            t_flag = 1
        w, h = image.size
        # newh = MAX_W
        # newh = h
        newh = 384
        # newh = max(384, h)
        resize_ratio = newh / h
        
        neww = math.ceil(w * resize_ratio)
        # neww = 384
        # print(f"{h}, {w} resize to {newh}, {neww}")
        try:
            image = image.resize((neww, newh), resample=Image.BICUBIC)
        except:
            print(im_id)

        W, H = image.size

        new_H = 16 * int(H / 16)
        new_W = 16 * int(W / 16)
        scale_factor_w = float(new_W) / w
        scale_factor_h = float(new_H) / h
        image = transforms.Resize((new_H, new_W))(image)
        image = TTensor(image)

        # normalize
        if True:
            try:
                image = Normalize(image)
            except:
                print(im_id)

        rects = list()
        if t_flag == 0:
            for bbox in bboxes:
                x1 = round(bbox[0][0] * scale_factor_w)
                y1 = round(bbox[0][1] * scale_factor_h)
                x2 = round(bbox[2][0] * scale_factor_w)
                y2 = round(bbox[2][1] * scale_factor_h)
                rects.append([y1, x1, y2, x2])
        else:
            for bbox in bboxes:
                y1 = round(bbox[0][0] * scale_factor_h)
                x1 = round(bbox[0][1] * scale_factor_w)
                y2 = round(bbox[2][0] * scale_factor_h)
                x2 = round(bbox[2][1] * scale_factor_w)
                rects.append([y1, x1, y2, x2])

        boxes = list()
        scale_x = []
        scale_y = []
        # cnt_id = 3
        cnt = 0
        w_all = 0
        h_all = 0
        for box in rects:
            cnt += 1
            # modify shots
            if cnt > 3:
                break
            # if cnt != cnt_id:
            #     continue
            box2 = [int(k) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            scale_x1 = torch.tensor((x2 - x1 + 1) / 384)
            scale_x.append(scale_x1)
            scale_y1 = torch.tensor((y2 - y1 + 1) / 384)
            scale_y.append(scale_y1)
            bbox = image[:, y1:y2 + 1, x1:x2 + 1]
            bbox = transforms.Resize((64, 64))(bbox)
            boxes.append(bbox.numpy())
            w_all = float(x2 - x1) + w_all
            h_all = float(y2 - y1) + w_all
        scale_xx = torch.stack(scale_x).unsqueeze(-1)
        scale_yy = torch.stack(scale_y).unsqueeze(-1)
        scale = torch.cat((scale_xx, scale_yy), dim=1)
        boxes = np.array(boxes)
        boxes = torch.Tensor(boxes)

        w_all = w_all / 60
        h_all = h_all / 60

        gt_map = np.zeros((new_H, new_W), dtype='float32')

        if t_flag == 1:
            for i in range(dots.shape[0]):
                ptx = float(dots[i][1])
                pty = float(dots[i][0])
                dots[i][0] = ptx * scale_factor_w
                dots[i][1] = pty * scale_factor_h
                gt_map[min(round(float(dots[i][1])), new_H - 1)][
                    min(round(float(dots[i][0])), new_W - 1)] = 1
        else:
            for i in range(dots.shape[0]):
                dots[i][0] = float(dots[i][0]) * scale_factor_w
                dots[i][1] = float(dots[i][1]) * scale_factor_h
                gt_map[min(round(float(dots[i][1])), new_H - 1)][
                    min(round(float(dots[i][0])), new_W - 1)] = 1

        # gt_map = generate_density_map(dots,(newh,neww))

        gt_map = torch.from_numpy(gt_map)
        # gt_map = gt_map * 60

        size = getsize(scale)
        sample = {'image': image, 'dots': dots.shape[0], 'boxes': boxes, 'pos': rects, 'scale': scale,
                  'gt_density': gt_map, 'id':im_id, 'size':size}
        return sample
    

class Val_Test_Specific_dataset(Dataset):
    def __init__(self,dataset:PlantDataset,category:str,local=False):
        self.category = category
        self.annotations = dataset.annotations
        try:
            self.img = dataset.data_split['val'][category]
        except:
            self.img = dataset.data_split['test'][category]
        self.img_dir = dataset.data_path
        # for img in self.img_dir:
        #     if not img in val_category:


    def __len__(self):
        return len(self.img)
    

    def __getitem__(self, idx):
        im_id = self.img[idx]
        anno = self.annotations[im_id]
        bboxes = anno['box_examples_coordinates']

        dots = np.array(anno['points'])
        try:
            image = Image.open('{}{}'.format(self.img_dir, im_id))
            if image.mode != 'RGB':
                image = image.convert('RGB')
                print(im_id)

            w, h = image.size
            image.load()
        except:
            print(im_id)
        t_flag = 0
        if w < h:
            image = image.transpose(Image.TRANSPOSE)
            t_flag = 1
        w, h = image.size
        # newh = MAX_W
        # newh = h
        newh = 384
        # newh = max(384, h)
        resize_ratio = newh / h
        
        neww = math.ceil(w * resize_ratio)
        # neww = 384
        # print(f"{h}, {w} resize to {newh}, {neww}")
        try:
            image = image.resize((neww, newh), resample=Image.BICUBIC)
        except:
            print(im_id)

        W, H = image.size

        new_H = 16 * int(H / 16)
        new_W = 16 * int(W / 16)
        scale_factor_w = float(new_W) / w
        scale_factor_h = float(new_H) / h
        image = transforms.Resize((new_H, new_W))(image)
        image = TTensor(image)

        # normalize
        if True:
            try:
                image = Normalize(image)
            except:
                print(im_id)

        rects = list()
        if t_flag == 0:
            for bbox in bboxes:
                x1 = round(bbox[0][0] * scale_factor_w)
                y1 = round(bbox[0][1] * scale_factor_h)
                x2 = round(bbox[2][0] * scale_factor_w)
                y2 = round(bbox[2][1] * scale_factor_h)
                rects.append([y1, x1, y2, x2])
        else:
            for bbox in bboxes:
                y1 = round(bbox[0][0] * scale_factor_h)
                x1 = round(bbox[0][1] * scale_factor_w)
                y2 = round(bbox[2][0] * scale_factor_h)
                x2 = round(bbox[2][1] * scale_factor_w)
                rects.append([y1, x1, y2, x2])

        boxes = list()
        scale_x = []
        scale_y = []
        # cnt_id = 3
        cnt = 0
        w_all = 0
        h_all = 0
        for box in rects:
            cnt += 1
            # modify shots
            if cnt > 3:
                break
            # if cnt != cnt_id:
            #     continue
            box2 = [int(k) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            scale_x1 = torch.tensor((x2 - x1 + 1) / new_W)
            scale_x.append(scale_x1)
            scale_y1 = torch.tensor((y2 - y1 + 1) / new_H)
            scale_y.append(scale_y1)
            bbox = image[:, y1:y2 + 1, x1:x2 + 1]
            bbox = transforms.Resize((64, 64))(bbox)
            boxes.append(bbox.numpy())
            w_all = float(x2 - x1) + w_all
            h_all = float(y2 - y1) + w_all
        scale_xx = torch.stack(scale_x).unsqueeze(-1)
        scale_yy = torch.stack(scale_y).unsqueeze(-1)
        scale = torch.cat((scale_xx, scale_yy), dim=1)
        boxes = np.array(boxes)
        boxes = torch.Tensor(boxes)

        w_all = w_all / 60
        h_all = h_all / 60

        gt_map = np.zeros((new_H, new_W), dtype='float32')

        if t_flag == 1:
            for i in range(dots.shape[0]):
                ptx = float(dots[i][1])
                pty = float(dots[i][0])
                dots[i][0] = ptx * scale_factor_w
                dots[i][1] = pty * scale_factor_h
                gt_map[min(round(float(dots[i][1])), new_H - 1)][
                    min(round(float(dots[i][0])), new_W - 1)] = 1
        else:
            for i in range(dots.shape[0]):
                dots[i][0] = float(dots[i][0]) * scale_factor_w
                dots[i][1] = float(dots[i][1]) * scale_factor_h
                gt_map[min(round(float(dots[i][1])), new_H - 1)][
                    min(round(float(dots[i][0])), new_W - 1)] = 1

        # gt_map = generate_density_map(dots,(newh,neww))

        gt_map = torch.from_numpy(gt_map)
        # gt_map = gt_map * 60
       
        sample = {'image': image, 'dots': dots.shape[0], 'boxes': boxes, 'pos': rects, 'scale': scale,
                  'gt_density': gt_map, 'id':im_id}
        return sample
    

class resizeTrainImage(object):
    """
    Resize the image so that:
        1. Image is equal to 384 * 384
        2. The new height and new width are divisible by 16
        3. The aspect ratio is possibly preserved
    Density map is cropped to have the same size(and position) with the cropped image
    Exemplar boxes may be outside the cropped area.
    Augmentation including Gaussian noise, Color jitter, Gaussian blur, Random affine, Random horizontal flip and Mosaic (or Random Crop if no Mosaic) is used.
    """
    
    def __init__(self, dataset:PlantDataset,MAX_HW=384,local:dict=False):
        self.dataset = dataset
        self.train_set = dataset.data_split['train']
        self.max_hw = MAX_HW
        self.local = local
        self.parameter = []
        if local['local']:
            for i in range(len(local['blocksize'])):
                self.parameter.append({'blocksize':local['blocksize'][i],
                                    'outputstride':local['output_stride'][i]
                                    })
    
    def random_scale(self, image, scale_range=(1.0, 1.5), ratio_range=(1.0, 1.2), interpolation=Image.BILINEAR):
        scale_factor = random.uniform(*scale_range)
        rat_p = random.random()
        if rat_p > 0.5:
            ratio_factor = random.uniform(*ratio_range)
        else:ratio_factor = 1.0
        new_width = int(image.width * scale_factor * ratio_factor)
        new_height = int(image.height * scale_factor)
        scaled_image = image.resize((new_width, new_height), interpolation)
        width_ratio = new_width / image.width
        height_ratio = new_height / image.height
        
        return scaled_image, new_width, new_height
    
                
    def __call__(self, sample):
        image, lines_boxes, density, dots, im_id, m_flag, idx = sample['image'], sample['lines_boxes'],sample['gt_density'], sample['dots'], sample['id'], sample['m_flag'], sample['idx']
        W, H = image.size
        scl_p = random.random()
        if scl_p > 0.5:
            image, new_W, new_H = self.random_scale(image=image)
            new_H = 16*int(new_H/16)
            new_W = 16*int(new_W/16)
        else:
            new_H = 16*int(H/16)
            new_W = 16*int(W/16)
        scale_factor_h = float(new_H)/ H
        scale_factor_w = float(new_W)/ W
        resized_image = transforms.Resize((new_H, new_W))(image)
        resized_density = cv2.resize(density, (new_W, new_H))

        # Augmentation probability
        aug_p = random.random()
        aug_flag = 0
        mosaic_flag = 0
        if aug_p < 0.4:
            aug_flag = 1
            if aug_p < 0.25:
                aug_flag = 0
                mosaic_flag = 1

        # Gaussian noise
        resized_image = TTensor(resized_image)
        if aug_flag == 1:
            noise = np.random.normal(0, 0.1, resized_image.size())
            noise = torch.from_numpy(noise)
            re_image = resized_image + noise
            re_image = torch.clamp(re_image, 0, 1)

            # Color jitter and Gaussian blur
            re_image = Augmentation(re_image)

            # Random horizontal flip
            flip_p = random.random()
            if flip_p > 0.5:
                resized_density = torch.tensor(resized_density)
                re_image = TF.hflip(re_image)
                resized_density = TF.hflip(resized_density).squeeze(0)
        
        # Random 384*384 crop in a new_W*384 image and 384*new_W density map
        if mosaic_flag == 0:
            if aug_flag == 0:
                re_image = resized_image
                resized_density = np.zeros((resized_density.shape[0], resized_density.shape[1]),dtype='float32')
                for i in range(dots.shape[0]):
                    resized_density[min(new_H-1,round(dots[i][1]*scale_factor_h))][min(new_W-1,round(dots[i][0]*scale_factor_w))]=1
                resized_density = torch.from_numpy(resized_density)
            
            
            start_w = random.randint(0, new_W-1-383)
            start_h = random.randint(0, new_H-1-383)
            reresized_image = TF.crop(re_image, start_h, start_w, 384, 384)
            reresized_density = resized_density[start_h:start_h+384, start_w:start_w+384]

        # Random self mosaic
        else:
            image_array = []
            map_array = []
            blending_l = random.randint(10, 20)
            resize_l = 192 + 2 * blending_l
            if dots.shape[0] >= 70:
                for i in range(4):
                    length =  random.randint(350, min(new_W, new_H))
                    start_W = random.randint(0, new_W-length)
                    start_H = random.randint(0, new_H-length)
                    reresized_image1 = TF.crop(resized_image, start_H, start_W, length, length)
                    reresized_image1 = transforms.Resize((resize_l, resize_l))(reresized_image1)
                    reresized_density1 = np.zeros((resize_l,resize_l),dtype='float32')
                    for i in range(dots.shape[0]):
                        if min(new_H-1,round(dots[i][1]*scale_factor_h)) >= start_H and min(new_H-1,round(dots[i][1]*scale_factor_h)) < start_H + length and min(new_W-1,round(dots[i][0]*scale_factor_w)) >= start_W and min(new_W-1,round(dots[i][0]*scale_factor_w)) < start_W + length:
                            reresized_density1[min(resize_l-1,round((min(new_H-1,round(dots[i][1]*scale_factor_h))-start_H)*resize_l/length))][min(resize_l-1,int((min(new_W-1,round(dots[i][0]*scale_factor_w))-start_W)*resize_l/length))]=1
                    reresized_density1 = torch.from_numpy(reresized_density1)
                    image_array.append(reresized_image1)
                    map_array.append(reresized_density1)
            else:
                m_flag = 1
                prob = random.random()
                if prob > 0.25:
                    gt_pos = random.randint(0,3)
                else:
                    gt_pos = random.randint(0,4) # 5% 0 objects
                for i in range(4):
                    if i == gt_pos:
                        Tim_id = im_id
                        r_image = resized_image
                        Tdots = dots
                        new_TH = new_H
                        new_TW = new_W
                        Tscale_factor_h = scale_factor_h
                        Tscale_factor_w = scale_factor_w
                    else:
                        Tim_id = self.train_set[random.randint(0, len(self.train_set)-1)]
                        Tdots = np.array(self.dataset.annotations[Tim_id]['points'])
                        '''while(abs(Tdots.shape[0]-dots.shape[0]<=10)):
                            Tim_id = train_set[random.randint(0, len(train_set)-1)]
                            Tdots = np.array(annotations[Tim_id]['points'])'''
                        Timage = Image.open('{}{}'.format(self.dataset.data_path, Tim_id))
                        TTw, TTh = Timage.size
                        t_flag = 0
                        if TTw < TTh:
                            Timage = Timage.transpose(Image.TRANSPOSE)
                            t_flag = 1
                        TTw, TTh = Timage.size
                        TTnewh = 384
                        resize_ratio_t = TTnewh / TTh
                        TTneww = math.ceil(TTw * resize_ratio_t)
                        Timage = Timage.resize((TTneww, TTnewh), resample=Image.BICUBIC)
                        if t_flag == 1:
                            for TTi in range(Tdots.shape[0]):
                                ptx = float(Tdots[TTi][1])
                                pty = float(Tdots[TTi][0])
                                Tdots[TTi][0] = ptx * resize_ratio_t
                                Tdots[TTi][1] = pty * resize_ratio_t
                        else:
                            for TTi in range(Tdots.shape[0]):
                                ptx = float(Tdots[TTi][0])
                                pty = float(Tdots[TTi][1])
                                Tdots[TTi][0] = ptx * resize_ratio_t
                                Tdots[TTi][1] = pty * resize_ratio_t
                        Timage.load()
                        new_TH = 16*int(Timage.size[1]/16)
                        new_TW = 16*int(Timage.size[0]/16)
                        Tscale_factor_w = float(new_TW) / Timage.size[0]
                        Tscale_factor_h = float(new_TH) / Timage.size[1]
                        r_image = TTensor(transforms.Resize((new_TH, new_TW))(Timage))

                    length =  random.randint(350, min(new_TW, new_TH))
                    start_W = random.randint(0, new_TW-length)
                    start_H = random.randint(0, new_TH-length)
                    reresize_ratio=length/resize_l
                    r_image1 = TF.crop(r_image, start_H, start_W, length, length)
                    r_image1 = transforms.Resize((resize_l, resize_l))(r_image1)
                    r_density1 = np.zeros((resize_l,resize_l),dtype='float32')
                    if self.dataset.class_dict[im_id] == self.dataset.class_dict[Tim_id]:
                        for i in range(Tdots.shape[0]):
                            if min(new_TH-1,round(Tdots[i][1]*Tscale_factor_h)) >= start_H and min(new_TH-1,round(Tdots[i][1]*Tscale_factor_h)) < start_H + length and min(new_TW-1,round(Tdots[i][0]*Tscale_factor_w)) >= start_W and min(new_TW-1,round(Tdots[i][0]*Tscale_factor_w)) < start_W + length:
                                r_density1[min(resize_l-1,int((min(new_TH-1,round(Tdots[i][1]*Tscale_factor_h))-start_H)*resize_l/length))][min(resize_l-1,int((min(new_TW-1,round(Tdots[i][0]*Tscale_factor_w))-start_W)*resize_l/length))]=1
                    r_density1 = torch.from_numpy(r_density1)
                    image_array.append(r_image1)
                    map_array.append(r_density1)


            reresized_image5 = torch.cat((image_array[0][:,blending_l:resize_l-blending_l],image_array[1][:,blending_l:resize_l-blending_l]),1)
            reresized_density5 = torch.cat((map_array[0][blending_l:resize_l-blending_l],map_array[1][blending_l:resize_l-blending_l]),0)
            for i in range(blending_l):
                    reresized_image5[:,192+i] = image_array[0][:,resize_l-1-blending_l+i] * (blending_l-i)/(2*blending_l) + reresized_image5[:,192+i] * (i+blending_l)/(2*blending_l)
                    reresized_image5[:,191-i] = image_array[1][:,blending_l-i] * (blending_l-i)/(2*blending_l) + reresized_image5[:,191-i] * (i+blending_l)/(2*blending_l)
            reresized_image5 = torch.clamp(reresized_image5, 0, 1)

            reresized_image6 = torch.cat((image_array[2][:,blending_l:resize_l-blending_l],image_array[3][:,blending_l:resize_l-blending_l]),1)
            reresized_density6 = torch.cat((map_array[2][blending_l:resize_l-blending_l],map_array[3][blending_l:resize_l-blending_l]),0)
            for i in range(blending_l):
                    reresized_image6[:,192+i] = image_array[2][:,resize_l-1-blending_l+i] * (blending_l-i)/(2*blending_l) + reresized_image6[:,192+i] * (i+blending_l)/(2*blending_l)
                    reresized_image6[:,191-i] = image_array[3][:,blending_l-i] * (blending_l-i)/(2*blending_l) + reresized_image6[:,191-i] * (i+blending_l)/(2*blending_l)
            reresized_image6 = torch.clamp(reresized_image6, 0, 1)

            reresized_image = torch.cat((reresized_image5[:,:,blending_l:resize_l-blending_l],reresized_image6[:,:,blending_l:resize_l-blending_l]),2)
            reresized_density = torch.cat((reresized_density5[:,blending_l:resize_l-blending_l],reresized_density6[:,blending_l:resize_l-blending_l]),1)
            for i in range(blending_l):
                    reresized_image[:,:,192+i] = reresized_image5[:,:,resize_l-1-blending_l+i] * (blending_l-i)/(2*blending_l) + reresized_image[:,:,192+i] * (i+blending_l)/(2*blending_l)
                    reresized_image[:,:,191-i] = reresized_image6[:,:,blending_l-i] * (blending_l-i)/(2*blending_l) + reresized_image[:,:,191-i] * (i+blending_l)/(2*blending_l)
            reresized_image = torch.clamp(reresized_image, 0, 1)

        # normalize
        if True: 
            reresized_image = Normalize(reresized_image)
            resized_image = Normalize(resized_image)

        # Crop bboxes and resize as 64x64
        boxes = list()
        scale_x = []
        scale_y = []
        cnt = 0
        w_all=0
        h_all=0
        for box in lines_boxes:
            cnt+=1
            if cnt>3:
                break
            box2 = [int(k) for k in box]
            y1, x1, y2, x2 = round(box2[0]*scale_factor_h), round(box2[1]*scale_factor_w), round(box2[2]*scale_factor_h), round(box2[3]*scale_factor_w)
            scale_x1 = torch.tensor((x2-x1+1))/384
            scale_y1 = torch.tensor((y2-y1+1))/384
            scale_x.append(scale_x1)
            scale_y.append(scale_y1)
            bbox = resized_image[:,y1:y2+1,x1:x2+1]
            bbox = transforms.Resize((64, 64))(bbox)
            boxes.append(bbox.numpy())
            w_all = float(x2 - x1) + w_all
            h_all = float(y2 - y1) + w_all
        scale_xx = torch.stack(scale_x).unsqueeze(-1)
        scale_yy = torch.stack(scale_y).unsqueeze(-1)
        scale = torch.cat((scale_xx,scale_yy),dim=1)

        # only for training 128 32 at the same time
        size = getsize(scale)
        size_embedding = torch.zeros(len(self.parameter))
        if len(self.parameter) == 1:
            size_embedding[0] = 1
        else:
            try:
                for iter, p in enumerate(self.parameter):
                    if size < p['blocksize']:
                        break
                size_embedding[iter] = 1
            except:
                size_embedding = 0

        if m_flag == 1:
            w_all = w_all/reresize_ratio
            h_all = h_all/reresize_ratio
        w_all = w_all / 60
        h_all = h_all / 60


        boxes = np.array(boxes)
        boxes = torch.Tensor(boxes)
        
        
        reresized_density = ndimage.gaussian_filter(reresized_density, sigma=(h_all, w_all), order=0)
        ori_density = reresized_density.copy()

        # Density map scale up
        reresized_density = torch.from_numpy(reresized_density)
        reresized_density = reresized_density.squeeze(0).squeeze(0)
        # boxes shape [3,3,64,64], image shape [3,384,384], density shape[384,384]

        reresized_image = reresized_image.to(torch.float32)
        boxes = boxes.to(torch.float32)
        reresized_density = reresized_density.to(torch.float32)

        # sample = {'image':reresized_image,'boxes':boxes,'gt_density':reresized_density, 'm_flag': m_flag,'scale':scale, 'id':im_id, 'dots':dots.shape[0]}
        sample = {'image':reresized_image,'boxes':boxes,'gt_density':reresized_density, 'm_flag': m_flag,'scale':scale, 'id':im_id, 'dots':dots.shape[0],
                  'ori_density':ori_density,
                  'size_embedding':size_embedding,
                  'size':size}
        return sample


TTensor = transforms.Compose([   
        transforms.ToTensor(),
        ])

Augmentation = transforms.Compose([   
        transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.15, hue=0.15),
        transforms.GaussianBlur(kernel_size=(7,9))
        ])

Normalize = transforms.Compose([   
        transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)
        ])


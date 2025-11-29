import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import math
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.lr_sched as lr_sched

from datasets.datasets import build_dataloader
from util.misc import plot_learning_curves
from models.build_model import build_model
from torch.optim.lr_scheduler import OneCycleLR

def build_criterion(**kwargs):
    if kwargs['loss'] == 'L1':
        criterion = nn.L1Loss(reduction='none')
    elif kwargs['loss'] == 'L2':
        criterion = nn.MSELoss(reduction='none')
    return [criterion, kwargs['use_mask']]



def main(config):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print(config)
    device = torch.device(config['Training']['device'])
    use_enhance_density = config['Training']['use_enhance_density']

    resume_path = config['Resume']['resume_path']

    # fix the seed for reproducibility
    seed = config['Training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # define dataloader
    data_loader_train = build_dataloader('train',**config['Dataset'])
    data_loader_val = build_dataloader('val',**config['Dataset'])

    # define model
    model = build_model(**config['Model'])
    model.to(device)
    # for key, m in model.counter.items():
    #     m.to(device)

    model_without_ddp = model
    name = config['Model']['name']
    print("Model = %s" % str(model_without_ddp))
    if config['Resume']['checkpoint']:
        misc.load_model_check(resume=resume_path, model_without_ddp=model_without_ddp)
    else:
        misc.load_model_FSC(resume=[name, resume_path], model_without_ddp=model_without_ddp)

    # define optimizer
    lr = config['Optimizer']['lr']
    blr = config['Optimizer']['blr']
    eff_batch_size = config['Dataset']['batch_size'] * config['Optimizer']['accum_iter']
    if lr is None:  # only base_lr is specified
        lr = blr * eff_batch_size / 256
    print("base lr: %.2e" % (lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % lr)
    print("accumulate grad iterations: %d" % config['Optimizer']['accum_iter'])
    print("effective batch size: %d" % eff_batch_size)
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, config['Optimizer']['weight_decay'])
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))

    epochs, start_epoch = config['Optimizer']['epochs'], config['Optimizer']['start_epoch']
    # optimizer = optimizer.to(device)
    print(optimizer)
    
    # define loss with amp
    loss_scaler = NativeScaler(multi=True)

    # define logger
    log_writer = None
    if config['logger']['log']:
        log_writer = SummaryWriter(config['logger']['log_path'])


    output_dir = config['Callback']['output_dir']
    print(f"Start training for {epochs} epochs")
    start_time = time.time()
    mae=10000000
    mse=10000000
    loss_list = []
    val_mae_list = []
    factor = config['Dataset']['factor']
    criterion, use_mask = build_criterion(**config['Loss'])
    criterion = criterion.to(device)

    # define local conv
    kernel_list, weight_list = [], []
    if config['Dataset']['local']['local']:
        for i in range(len(config['Dataset']['local']['blocksize'])):
            kernel = torch.ones((config['Dataset']['local']['blocksize'][i], config['Dataset']['local']['blocksize'][i]))
            kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).to(device)
            kernel_list.append(kernel)
            weight = nn.Parameter(data=kernel, requires_grad=False).to(device).to(torch.float16)
            weight_list.append({'blocksize':config['Dataset']['local']['blocksize'][i],
                                'outputstride':config['Dataset']['local']['output_stride'][i],
                                'weight':weight
                                })

    from val_local import val_func_multihead
    # (optianl) santiy check
    # mae_test,mse_test,wca, r2_test = val_func_multihead(model=model,
    #                                                dataloader=data_loader_val,
    #                                                config=config, 
    #                                                weight_list=weight_list)
    for epoch in range(start_epoch,epochs):
        # train one epoch
        model.train(True)
        accum_iter = config['Optimizer']['accum_iter']

        # some parameters in training
        train_mae = 0
        train_rmse = 0
        pred_cnt = 0
        gt_cnt = 0
        loss_per_epoch = 0

        optimizer.zero_grad()

        if log_writer is not None:
            print('log_dir: {}'.format(log_writer.log_dir))
        
        with tqdm(data_loader_train, dynamic_ncols=True) as pbar:
            for data_iter_step, samples in enumerate(pbar):
                if data_iter_step % accum_iter == 0:
                    lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_train) + epoch, **config['Optimizer'])
                image = samples['image'].to(device, non_blocking=True).half()
                B,C,H_ori,W_ori = image.shape
                gt_density = samples['gt_density'].to(device, non_blocking=True).half()
                boxes = samples['boxes'].to(device, non_blocking=True).half()
                scales = samples['scale'].to(device, non_blocking=True).half()
                size_embedding = samples['size_embedding'].to(device, non_blocking=True).half()
                size = samples['size'].to(device, non_blocking=True).half()
                shot_num = 3
                boxes = boxes[:,:shot_num,:,:,:]
                scales = scales[:,:shot_num,:]
                with torch.cuda.amp.autocast():
                    inputx = [image,boxes,scales,size]
                    outputs = model(inputx,test=False)

                batch_mae = 0
                batch_rmse = 0
                pred_cnt_list = []
                gt_cnt_list = []
                output_list = []
                lr = optimizer.param_groups[0]["lr"]

                loss_train_list = []
                # Compute loss function
                if isinstance(outputs, dict):
                    if not use_enhance_density:
                        for iter, p in enumerate(weight_list):
                            bl, ops, weight = p['blocksize'], p['outputstride'], p['weight']
                            output = outputs[f'{bl}']
                            H, W = output.shape[-2:]
                            if use_mask:
                                mask = np.random.binomial(n=1, p=0.8, size=[H,W])
                                masks = np.tile(mask,(output.shape[0],1))
                                masks = masks.reshape(output.shape[0], H, W)
                                masks = torch.from_numpy(masks).to(device)
                            else:
                                mask = np.ones(shape=(output.shape[0],H,W))
                                masks = torch.from_numpy(mask).to(device)
                            loss = criterion(output * size_embedding[:,iter].reshape(B,1,1),
                                            (F.conv2d(gt_density.reshape(B,1,H_ori,W_ori), weight, stride=ops)).reshape(B,H,W) * size_embedding[:,iter].reshape(B,1,1))
                            loss = loss.type(torch.float64)
                            loss = (loss * masks / (H*W)).sum()
                            loss_value = loss.item()
                            loss_train_list.append(loss)

                            loss_per_epoch += loss_value
                    else:
                        loss = torch.zeros(size=(B,1)).to(torch.float64).to('cuda')
                        for iter, p in enumerate(weight_list):
                            bl, ops, weight = p['blocksize'], p['outputstride'], p['weight']
                            if not f'{bl}' in list(outputs.keys()):
                                continue
                            output = outputs[f'{bl}']
                            H, W = output.shape[-2:]
                            loss += criterion(output.sum(dim=(1,2,3)).unsqueeze(-1),
                                            (((F.conv2d(gt_density.reshape(B,1,H_ori,W_ori), weight, stride=ops)).reshape(B,H,W))).to(torch.float32).sum(dim=(1,2)).unsqueeze(-1))
                        
                            # loss_per_epoch += loss_value
                            batch_mae = (output.sum(dim=(1,2,3)).unsqueeze(-1) - F.conv2d(gt_density.reshape(B,1,H_ori,W_ori), weight, stride=ops).reshape(B,H,W).to(torch.float32).sum(dim=(1,2)).unsqueeze(-1)).sum().item()
                            batch_rmse += batch_mae ** 2

                        loss = (loss / (H*W)).sum() / output.shape[0]
                        loss_value = loss.item()
                        loss_per_epoch += loss_value



                    pred_cnt = 0
                    for iter, level in enumerate(outputs):
                        pred_cnt += torch.sum(outputs[level] * size_embedding[:,iter].reshape(B,1,1)).item()
                    gt_cnt = torch.sum(gt_density/factor).item()
                    cnt_err = pred_cnt - gt_cnt
                    batch_mae += cnt_err
                    batch_rmse += cnt_err ** 2

                    train_mae += batch_mae
                    train_rmse += batch_rmse

                    if not math.isfinite(loss_value):
                        print("Loss is {}, stopping training".format(loss_value))
                        sys.exit(1)

                    loss /= accum_iter
                    loss_scaler(loss_train_list, optimizer, parameters=model.parameters(),
                                update_grad=(data_iter_step + 1) % accum_iter == 0)
                    if (data_iter_step + 1) % accum_iter == 0:
                        optimizer.zero_grad()
                    
                    torch.cuda.synchronize(device)
                    pbar.set_description(f"Epoch: {epoch}")
                else:
                    H, W = outputs.shape[-2:]
                    if use_mask:
                        mask = np.random.binomial(n=1, p=0.8, size=[H,W])
                        masks = np.tile(mask,(outputs.shape[0],1))
                        masks = masks.reshape(outputs.shape[0], H, W)
                        masks = torch.from_numpy(masks).to(device)
                    else:
                        mask = np.ones(shape=(outputs.shape[0],H,W))
                        masks = torch.from_numpy(mask).to(device)
                    loss = criterion(outputs,gt_density)
                    loss = loss.type(torch.float64)
                    loss = (loss * masks / (H*W)).sum() / outputs.shape[0]
                    loss_value = loss.item()
                    loss_per_epoch += loss_value * outputs.shape[0]

                    batch_mae = 0
                    batch_rmse = 0
                    pred_cnt_list = []
                    gt_cnt_list = []
                    output_list = []
                    lr = optimizer.param_groups[0]["lr"]

                    for i in range(outputs.shape[0]):
                        pred_cnt = 0
                        pred_cnt = torch.sum(outputs[i]/factor).item()
                        pred_cnt_list.append(pred_cnt)
                        output_list.append(outputs[i])
                        gt_cnt = torch.sum(gt_density[i]/factor).item()
                        gt_cnt_list.append(gt_cnt)
                        cnt_err = abs(pred_cnt - gt_cnt)
                        batch_mae += cnt_err
                        batch_rmse += cnt_err ** 2

                    train_mae += batch_mae
                    train_rmse += batch_rmse

                    if not math.isfinite(loss_value):
                        print("Loss is {}, stopping training".format(loss_value))
                        sys.exit(1)

                    loss /= accum_iter
                    loss_scaler(loss, optimizer, parameters=model.parameters(),
                                update_grad=(data_iter_step + 1) % accum_iter == 0)
                    if (data_iter_step + 1) % accum_iter == 0:
                        optimizer.zero_grad()
                    
                    torch.cuda.synchronize(device)
                    pbar.set_description(f"Epoch: {epoch}")

        # write logger per epoch
        loss_per_epoch = loss_per_epoch/len(data_loader_train.dataset)
        if log_writer is not None:
            log_writer.add_scalar('train_loss_per_epoch_avg', loss_per_epoch, epoch)
            log_writer.add_scalar('lr', lr, epoch)
            log_writer.add_scalar('MAE', train_mae/len(data_loader_train.dataset), epoch)
            log_writer.add_scalar('RMSE', (train_rmse/len(data_loader_train.dataset))**0.5, epoch)

        # save train status and model
        if output_dir and (epoch % 50 == 0 or epoch + 1 == epochs):
            misc.save_model(output_dir=output_dir, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,loss_scaler=loss_scaler, epoch=epoch)
            config['Optimizer']['start_epoch'] = epoch + 1
            misc.save_yaml(output_dir,config)

        # val and save val log
        if output_dir and epoch >= 50:
            mae_new,mse_new,wca_new,r2_new  = val_func_multihead(model=model,
                                                dataloader=data_loader_val,
                                                config=config, 
                                                weight_list=weight_list)
            # if better, val on test set
            if mae_new<mae:
                mae = mae_new
                mse = mse_new
                misc.save_model(output_dir=output_dir, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,loss_scaler=loss_scaler, epoch=epoch+1)
            log_stats = {'VAL MAE': mae_new,
                         'VAL RMSE': mse_new,
                         'VAL WCA':wca_new,
                         'VAL R2': r2_new,
                         'BEST MAE': mae,
                         'BEST MSE': mse,
                         'epoch': epoch}
            val_mae_list.append(mae_new)
            if output_dir:
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(output_dir, "log_val.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
        loss_list.append(loss_per_epoch)
        plot_learning_curves(loss_list, val_mae_list, output_dir)

        # Output training log status
        log_stats = {'lr': lr,
                     'loss': loss_per_epoch,
                    'Current MAE': abs(train_mae)/(len(data_loader_train.dataset)),
                    'RMSE':  (train_rmse/(len(data_loader_train.dataset)))**0.5,
                    'epoch': epoch}
        print('Current MAE: {:5.2f}, RMSE: {:5.2f} '.format( train_mae/(len(data_loader_train.dataset)), (train_rmse/(len(data_loader_train.dataset)))**0.5))
        if output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    config = misc.load_yaml("cfg_train/local_32_64_128_loose_revised.yml", encoding="utf-8")
    if config['Callback']['output_dir']:
        Path(config['Callback']['output_dir']).mkdir(parents=True, exist_ok=True)
    main(config)

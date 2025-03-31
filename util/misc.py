# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import matplotlib.patches as mpatches
import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch import inf
import torch.nn as nn
from sklearn.linear_model import LinearRegression
import numpy as np
from typing import Union
from pathlib2 import Path
import yaml
import cv2
from matplotlib.colors import ListedColormap, BoundaryNorm

def plot_learning_curves(loss, val_mae, dir_to_save):
    # plot learning curves
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(loss, label='train loss', color='tab:blue')
    ax1.legend(loc = 'upper right')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(val_mae, label='val mae', color='tab:orange')
    ax2.legend(loc = 'upper right')
    # ax2.set_ylim((0,50))
    fig.savefig(os.path.join(dir_to_save, 'learning_curves.png'), bbox_inches='tight', dpi = 300)
    plt.close()


class SumConv2d(nn.Module):
    def __init__(self):
        super(SumConv2d, self).__init__()
        kernel_size = 16
        stride = 16
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=0, bias=False)
        self.conv.weight.data.fill_(1)
        # 禁止权重更新
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 禁止梯度计算
        x = x.unsqueeze(1)
        with torch.no_grad():
            return self.conv(x).squeeze()

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, multi=True):
        self._scaler = torch.cuda.amp.GradScaler(init_scale=2.0**14)
        self.multi = multi

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        for i in range(len(loss)):
            if i == len(loss) - 1:
                self._scaler.scale(loss[i]).backward()
            else:
                self._scaler.scale(loss[i]).backward(create_graph=create_graph,retain_graph=True)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
            return norm  

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(output_dir, epoch, model, model_without_ddp, optimizer, loss_scaler):
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [os.path.join(output_dir, f'checkpoint-{epoch_name}.pth')]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)

def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != model_without_ddp.state_dict()['pos_embed'].shape:
            print(f"Removing key pos_embed from pretrained checkpoint")
            del checkpoint['model']['pos_embed']

        if 'decoder_pos_embed' in checkpoint['model'] and checkpoint['model']['decoder_pos_embed'].shape != model_without_ddp.state_dict()['decoder_pos_embed'].shape:
            print(f"Removing key decoder_pos_embed from pretrained checkpoint")
            del checkpoint['model']['decoder_pos_embed']
            
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")

def load_model_finetune(resume, model_without_ddp, check=True):
    if check:
        checkpoint = torch.load(resume, map_location='cpu')
        if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != model_without_ddp.state_dict()['pos_embed'].shape:
            print(f"Removing key pos_embed from pretrained checkpoint")
            del checkpoint['model']['pos_embed']
        for key in list(checkpoint['model'].keys()):
            print(f"checkpoint:{key} {checkpoint['model'][key].shape}")
        for name, p in model_without_ddp.named_parameters():
            print(f"model:{name} {p.shape}")

    model_without_ddp.load_state_dict(checkpoint['model'],strict=False)
    for name, p in model_without_ddp.named_parameters():
        if not 'density_regress' in name:
            p.requires_grad = False

def load_model_FSC(resume:list, model_without_ddp, check=True):
    if resume[1]:
        if check:
            checkpoint = torch.load(resume[1], map_location='cpu')
            for key in list(checkpoint['model'].keys()):
                print(f"checkpoint:{key} {checkpoint['model'][key].shape}")
            for name, p in model_without_ddp.named_parameters():
                print(f"model:{name} {p.shape}")

        if resume[1].startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                resume[1], map_location='cpu', check_hash=True)
            
        elif resume[0] == 'mae_vit_base_patch16_dec512d8b' or resume[0] == 'local_with_decoder':
            checkpoint = torch.load(resume[1], map_location='cpu')
            if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != model_without_ddp.state_dict()['pos_embed'].shape:
                print(f"Removing key pos_embed from pretrained checkpoint")
                del checkpoint['model']['pos_embed']
            if 'decoder_pos_embed' in checkpoint['model'] and checkpoint['model']['decoder_pos_embed'].shape != model_without_ddp.state_dict()['decoder_pos_embed'].shape:
                print(f"Removing key decoder_pos_embed from pretrained checkpoint")
                del checkpoint['model']['decoder_pos_embed']
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

        elif resume[0] == 'encoder_only' or resume[0] == 'local_count':
            checkpoint = torch.load(resume[1],map_location='cpu')
            if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != model_without_ddp.state_dict()['pos_embed'].shape:
                print(f"Removing key pos_embed from pretrained checkpoint")
                del checkpoint['model']['pos_embed']
            key_map = []
            for key,item in checkpoint['model'].items():
                if 'decode' in key:
                    key_map.append(key)
            for key in key_map:
                del checkpoint['model'][key]
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        else:
            if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != model_without_ddp.state_dict()['pos_embed'].shape:
                print(f"Removing key pos_embed from pretrained checkpoint")
                del checkpoint['model']['pos_embed']
                model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Resume checkpoint %s" % resume[1])

def load_model_check(resume, model_without_ddp, check=True):
    if check:
        checkpoint = torch.load(resume, map_location='cpu')
        for key in list(checkpoint['model'].keys()):
            print(f"checkpoint:{key} {checkpoint['model'][key].shape}")
        for name, p in model_without_ddp.named_parameters():
            print(f"model:{name} {p.shape}")
    model_without_ddp.load_state_dict(checkpoint['model'],strict=True)

def load_model_FSC_encoder(args, model_without_ddp):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != model_without_ddp.state_dict()['pos_embed'].shape:
            print(f"Removing key pos_embed from pretrained checkpoint")
            del checkpoint['model']['pos_embed']
        
        # if 'decoder_pos_embed' in checkpoint['model'] and checkpoint['model']['decoder_pos_embed'].shape != model_without_ddp.state_dict()['decoder_pos_embed'].shape:
        #     print(f"Removing key decoder_pos_embed from pretrained checkpoint")
        #     del checkpoint['model']['decoder_pos_embed']
            
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Resume checkpoint %s" % args.resume)


def load_model_FSC_one_stage(args, model_without_ddp):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != model_without_ddp.state_dict()['pos_embed'].shape:
            print(f"Removing key pos_embed from pretrained checkpoint")
            del checkpoint['model']['pos_embed']
        
        if 'decoder_pos_embed' in checkpoint['model'] and checkpoint['model']['decoder_pos_embed'].shape != model_without_ddp.state_dict()['decoder_pos_embed'].shape:
            print(f"Removing key decoder_pos_embed from pretrained checkpoint")
            del checkpoint['model']['decoder_pos_embed']
            
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Resume checkpoint %s" % args.resume)

def load_model_FSC1(args, model_without_ddp):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            #model = timm.create_model('vit_base_patch16_224', pretrained=True)
            #torch.save(model.state_dict(), './output_abnopre_dir/checkpoint-6657.pth')
            checkpoint1 = torch.load('./output_abnopre_dir/checkpoint-6657.pth', map_location='cpu')

        if 'pos_embed' in checkpoint['model'] and checkpoint['model']['pos_embed'].shape != model_without_ddp.state_dict()['pos_embed'].shape:
            print(f"Removing key pos_embed from pretrained checkpoint")
            del checkpoint['model']['pos_embed']

        del checkpoint1['cls_token'],checkpoint1['pos_embed']
            
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        model_without_ddp.load_state_dict(checkpoint1, strict=False)
        print("Resume checkpoint %s" % args.resume)

def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

def add_weight_decay_lr(model,lr_back, weight_decay=1e-5, skip_list=()):
    decay = []
    decay_backbone = []
    no_decay = []
    no_decay_backbone = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            if 'blocks' in name and 'decoder' not in name:
                no_decay_backbone.append(param)
            else:
                no_decay.append(param)
        else:
            if 'blocks' in name and 'decoder' not in name: 
                decay_backbone.append(param)
            else:
                decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': no_decay_backbone, 'weight_decay': 0.,'lr':lr_back},
        {'params': decay, 'weight_decay': weight_decay},
        {'params': decay_backbone, 'weight_decay': weight_decay, 'lr':lr_back}]


def plot_scatter(gt_array,pred_array):
    plt.scatter(gt_array, pred_array)
    plt.xlabel('Ground Truth')
    plt.ylabel('Error')
    plt.savefig(f'./Image/test_stat_scatter.png')
    plt.show()
    plt.close()

    plt.scatter(gt_array, pred_array, label='Data', color='blue')
    model = LinearRegression()
    model.fit(np.array(gt_array).reshape(-1,1), np.array(pred_array))
    pred_line = model.predict(np.array(gt_array).reshape(-1, 1))

    plt.plot(gt_array, pred_line, color='red', label=f'Pred = {model.coef_[0]:.2f}*GT + {model.intercept_:.2f}')

    correlation_coefficient = np.corrcoef(gt_array, pred_array)[0, 1]

    plt.legend()
    plt.title('GT vs Pred Scatter Plot')
    plt.xlabel('GT')
    plt.ylabel('Pred')
    plt.annotate(f'Correlation Coefficient: {correlation_coefficient:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12)
    plt.savefig(f'./Image/test_fintuned_scatter.png')
    plt.show()    
    plt.close()

def load_yaml(yml_path: Union[Path, str], encoding="utf-8"):
    if isinstance(yml_path, str):
        yml_path = Path(yml_path)
    with yml_path.open('r', encoding=encoding) as f:
        cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return cfg
    
def save_yaml(yml_path: Union[Path, str], data, ):
    if not 'yaml' in yml_path:
        yml_path = os.path.join(yml_path, 'config.yaml')
    with open(yml_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)


import torch.nn.functional as F
def ZeroPadding(images:torch.Tensor, target_shape:tuple, return_size=False):
    try:
        bs, c, h, w = images.shape
    except:
        bs, h, w = images.shape
    target_h, target_w = target_shape

    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)

    if pad_h == 0 and pad_w == 0:
        if not return_size:
            return images
        else:
            return images, 0, 0, 0, 0

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padded_images = F.pad(images, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    if not return_size:
        return padded_images
    else:
        return padded_images,pad_left,pad_right,pad_top,pad_bottom


IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]
def denormalize(image:np.ndarray, mean=IM_NORM_MEAN, std=IM_NORM_STD):
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    image = image * std + mean 
    return image
def visualize_single_image_local(image,pre_density, gt_density, lc_map ,save_path=None, pred_cnt=None):
    if len(image.shape) == 4:
        image = image.squeeze(0)
    if len(pre_density.shape) != 3:
        h, w = pre_density.shape[-2:]
        pre_density = pre_density.reshape(h,w)
    if len(gt_density.shape) != 3:
        gt_density = gt_density.reshape(1,h,w)
    if not isinstance(gt_density,np.ndarray):
        gt_density = gt_density.detach().cpu().numpy()
    if not isinstance(pre_density,np.ndarray):
        pre_density = pre_density.detach().cpu().numpy()
    if not isinstance(image,np.ndarray):
        original_image = image.detach().cpu().numpy()
    original_image = denormalize(original_image)

    if original_image.shape[0] == 3:
        original_image = np.transpose(original_image, (1, 2, 0))

    fig = plt.figure(figsize=(12,8))

    ax = fig.add_subplot(1,4,1)
    ax.set_axis_off()
    ax.imshow(original_image)
    # legend_patch = mpatches.Patch(color='none', label="Image")
    # ax.legend(handles=[legend_patch], loc="upper right", handlelength=0, handletextpad=0) 

    ax = fig.add_subplot(1,4,2)
    ax.set_axis_off()
    ax.set_title(f"gt:{gt_density.sum()}")
    ax.imshow(original_image)
    ax.imshow(gt_density.squeeze(0), cmap='jet', alpha=0.4)
    # legend_patch = mpatches.Patch(color='none', label=f"gt:{gt_density.sum()}")
    # ax.legend(handles=[legend_patch], loc="upper right", handlelength=0, handletextpad=0) 

    ax = fig.add_subplot(1,4,3)
    ax.set_axis_off()

    # temp = pre_density
    # attention_map = cv2.normalize(temp , None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # attention_map = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    # ax.imshow(attention_map)
    # original_image = F.interpolate(original_image.unsqeeze(0),size=(h,w),mode='bilinear',align_corners=False)
    # color = [(1,1,1,0),
    #          (226/255,145/255,, 0.4)]
    # custom_cmap = ListedColormap(color)
    # bounds = [0, 0.5, 1]  # 分界点：0 �?? 1
    # norm = BoundaryNorm(bounds, len(color))  # 标准化到对应颜色

    ax.set_title(f"{pred_cnt:5.2f}")
    ax.imshow(original_image)
    ax.imshow(pre_density*0.6, cmap='viridis', alpha=0.4)
    # ax.imshow(pre_density, cmap=custom_cmap, norm=norm)
    # legend_patch = mpatches.Patch(color='none', label=f"pred:{pred_cnt:5.2f}")
    # ax.legend(handles=[legend_patch], loc="upper right", handlelength=0, handletextpad=0) 
    ax = fig.add_subplot(1,4,4)
    ax.set_axis_off()
    # ax.imshow(original_image)
    ax.imshow(lc_map.detach().cpu().numpy(), cmap='jet')

    

    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(save_path, bbox_inches='tight')

def visualize_single_image(image,pre_density, gt_density ,save_path=None):
    if len(image.shape) == 4:
        image = image.squeeze(0)
    if len(pre_density.shape) != 3:
        h, w = pre_density.shape[-2:]
        pre_density = pre_density.reshape(h,w)
    if len(gt_density.shape) != 3:
        gt_density = gt_density.reshape(1,h,w)
    if not isinstance(gt_density,np.ndarray):
        gt_density = gt_density.detach().cpu().numpy()
    if not isinstance(pre_density,np.ndarray):
        pre_density = pre_density.detach().cpu().numpy()
    if not isinstance(image,np.ndarray):
        original_image = image.detach().cpu().numpy()
    original_image = denormalize(original_image)

    if original_image.shape[0] == 3:
        original_image = np.transpose(original_image, (1, 2, 0))

    fig = plt.figure(figsize=(12,12))

    ax = fig.add_subplot(1,4,1)
    ax.set_axis_off()
    ax.imshow(original_image)
    legend_patch = mpatches.Patch(color='none', label="Image")
    ax.legend(handles=[legend_patch], loc="upper right", handlelength=0, handletextpad=0) 

    ax = fig.add_subplot(1,4,2)
    ax.set_axis_off()
    ax.imshow(original_image)
    ax.imshow(gt_density.squeeze(0), cmap='jet', alpha=0.6)
    # legend_patch = mpatches.Patch(color='blue', label=f"gt:{gt_density.sum()}")
    # ax.legend(handles=[legend_patch], loc="upper right", handlelength=0, handletextpad=0) 

    ax = fig.add_subplot(1,4,3)
    ax.set_axis_off()

    temp = pre_density
    attention_map = cv2.normalize(temp , None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    attention_map = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    ax.imshow(attention_map)

    ax = fig.add_subplot(1,4,4)
    ax.set_axis_off()
    ax.imshow(original_image)
    ax.imshow(pre_density, cmap='jet', alpha=0.6)
    # legend_patch = mpatches.Patch(color='blue', label="attn")
    # ax.legend(handles=[legend_patch], loc="upper right", handlelength=0, handletextpad=0) 


    

    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(save_path, bbox_inches='tight')

def visualize_single_image_local_other(image,pre_density, lc_map ,save_path=None, pred_cnt=None):
    if len(image.shape) == 4:
        image = image.squeeze(0)
    if len(pre_density.shape) != 3:
        h, w = pre_density.shape[-2:]
        pre_density = pre_density.reshape(h,w)
    if not isinstance(pre_density,np.ndarray):
        pre_density = pre_density.detach().cpu().numpy()
    if not isinstance(image,np.ndarray):
        original_image = image.detach().cpu().numpy()
    original_image = denormalize(original_image)

    if original_image.shape[0] == 3:
        original_image = np.transpose(original_image, (1, 2, 0))

    fig = plt.figure(figsize=(48,32))

    ax = fig.add_subplot(1,2,1)
    ax.set_axis_off()
    ax.imshow(original_image)
    # legend_patch = mpatches.Patch(color='none', label="Image")
    # ax.legend(handles=[legend_patch], loc="upper right", handlelength=0, handletextpad=0) 

    ax = fig.add_subplot(1,2,2)
    ax.set_axis_off()
    ax.set_title(f"{pred_cnt:5.2f}")
    ax.imshow(original_image)
    ax.imshow(pre_density*0.6, cmap='viridis', alpha=0.4)
    # ax.imshow(pre_density, cmap=custom_cmap, norm=norm)
    # legend_patch = mpatches.Patch(color='none', label=f"pred:{pred_cnt:5.2f}")
    # ax.legend(handles=[legend_patch], loc="upper right", handlelength=0, handletextpad=0) 
    # ax = fig.add_subplot(1,3,3)
    # ax.set_axis_off()
    # # ax.imshow(original_image)
    # ax.imshow(lc_map.detach().cpu().numpy(), cmap='jet')

    

    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(save_path, bbox_inches='tight')

def visual_attn(atten, atten_list,save_path=None):
    from torchvision import transforms
    amatch = atten.reshape(24,24)
    atten_map = torch.mean(atten_list[-1],dim=0).reshape(624, 624)
    fig = plt.figure(figsize=(12,12))


    # cmap = 'seismic'
    cmap = 'viridis' ## 密度图用的颜�??

    A_Query, A_Match, A_class, A_exp = extract_atten(atten_map)

    atten_map = atten_map.cpu().clone().numpy()
    atten_map = 255 * (atten_map / atten_map.max())
    atten_map = atten_map.astype(np.uint8)
    ax = fig.add_subplot(2,3,1)
    ax.set_axis_off()
    ax.imshow(atten_map,cmap=cmap)

    A_Query = A_Query.cpu().clone().numpy()
    A_Query = 255 * (A_Query / A_Query.max())
    A_Query = A_Query.astype(np.uint8)
    ax = fig.add_subplot(2,3,2)
    ax.set_axis_off()
    ax.imshow(A_Query,cmap=cmap)



    A_Match = A_Match.cpu().clone().numpy()
    A_Match = 255 * (A_Match / A_Match.max())
    A_Match = A_Match.astype(np.uint8)
    ax = fig.add_subplot(2,3,3)
    ax.set_axis_off()
    ax.imshow(A_Match,cmap=cmap)

    A_class = A_class.cpu().clone().numpy()
    A_class = 255 * (A_class / A_class.max())
    A_class = A_class.astype(np.uint8)
    ax = fig.add_subplot(2,3,4)
    ax.set_axis_off()
    ax.imshow(A_class,cmap=cmap)

    A_exp = A_exp.cpu().clone().numpy()
    A_exp = 255 * (A_exp / A_exp.max())
    A_exp = A_exp.astype(np.uint8)
    ax = fig.add_subplot(2,3,5)
    ax.set_axis_off()
    ax.imshow(A_exp,cmap=cmap)

    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(save_path, bbox_inches='tight')

def extract_atten(atten,ori_size=(576,576)):
    h, w = ori_size
    A_query = atten[:h,:w]
    A_math = atten[h:,:w]
    A_class = atten[:h,w:]
    A_exp = atten[h:,w:]
    # return atten_reshape(A_query), atten_reshape(A_math), atten_reshape(A_class), atten_reshape(A_exp)
    return atten_reshape(A_query), atten_reshape(A_math), atten_reshape(A_class), atten_reshape(A_exp)

from einops import rearrange,repeat
import math
def atten_reshape(atten):
    h,w = atten.shape
    if h==w and h==48:
        atten = atten.sum(dim=1).reshape(h,1).permute(1,0)
        atten = rearrange(atten,'b (n l)->b n l',n=3)
        atten = atten.sum(dim=1).reshape(1,16)
        atten = rearrange(atten,'b (h w)->b h w',w=4)
    elif h<=w:
        atten = atten.sum(dim=0).reshape(1,w)
        atten = rearrange(atten,'b (h w)->b h w',w=int(math.sqrt(w)))
    else:
        atten = atten.sum(dim=1).reshape(h,1).permute(1,0)
        atten = rearrange(atten,'b (h w)->b h w',w=int(math.sqrt(h)))
    return atten.squeeze(0)
import math
import torch
import numpy as np
from sklearn.metrics import r2_score
import os
from util.misc import ZeroPadding,visualize_single_image_local
import torch.nn.functional as F
import scipy.ndimage as ndimage
def val_func_multihead(model, dataloader, config, vis=False):
    # test
    model.eval()
    # some parameters in training
    train_mae = 0
    train_rmse = 0
    pred_cnt = 0
    gt_cnt = 0

    loss_array = []
    gt_array, pred_array = [], []
    wrong_id = []

    device = config['Training']['device']
    loss_value = 0
    scale_list_x, scale_list_y = [], []
    scale_list=[]

    for data_iter_step, samples in enumerate(dataloader):
        image = samples['image'].to(device, non_blocking=True)
        gt_dots = samples['dots'].to(device, non_blocking=True)
        boxes = samples['boxes'].to(device, non_blocking=True)
        scale = samples['scale'].to(device, non_blocking=True)
        im_id = samples['id']
        size = samples['size']


        _, _, h, w = image.shape
        pred_cnt = 0
        neww = math.ceil(w / h) * 384

        target_size = (384, neww)
        image,pl,pr,pu,pb = ZeroPadding(image, target_size, return_size=True)
        gt_density = ZeroPadding(samples['gt_density'],target_size)
        h_times = target_size[0] // 384
        w_times = target_size[1] // 384
        gt_cnt = 0
        merge_attn_list = []
        output_list = []
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for j in range(h_times):
                    for i in range(w_times):
                        input_x = [image[:, :, 384 * j:(384 + 384 * j), i * 384:(i * 384 + 384)], boxes, scale, size]
                        output, atten_map = model(input_x, size)
                        merge_attn_list.append(atten_map.squeeze(0).squeeze(0))
                        if isinstance(output, dict):
                            pred_cnt = 0
                            for k in output:
                                pred_cnt += (output[k]).sum().item()
                        else:
                            output_list.append(output.squeeze(0).squeeze(0))
                            pred_cnt += (output).sum().item()
                        gt_cnt += gt_density[:,384 * j:(384 + 384 * j),i * 384:(i * 384 + 384)].sum().item()
                    if target_size[0] % 384 != 0:
                        input_x = [image[:, :, 384 * j:(384 + 384 * j), -384:], boxes, scale, size]
                        output, atten_map = model(input_x, size)
                        merge_attn_list.append(atten_map.squeeze(0).squeeze(0))
                        if isinstance(output, dict):
                            pred_cnt = 0
                            for k in output:
                                pred_cnt += (output[k]).sum().item()
                        else:
                            output_list.append(output.squeeze(0).squeeze(0))
                            pred_cnt += (output).sum().item()
                        gt_cnt += gt_density[:,384 * j:(384 + 384 * j), -(w - w_times * 384):].sum().item()
                if target_size[1] % 384 != 0:
                    for i in range(w_times):
                        input_x = [image[:, :, -384:, i * 384:(i * 384 + 384)], boxes, scale, size]
                        output, atten_map = model(input_x, size)
                        merge_attn_list.append(atten_map.squeeze(0).squeeze(0))
                        if isinstance(output, dict):
                            pred_cnt = 0
                            for k in output:
                                pred_cnt += (output[k]).sum().item()
                        else:
                            output_list.append(output.squeeze(0).squeeze(0))
                            pred_cnt += (output).sum().item()
                        gt_cnt += gt_density[:,-(h - h_times * 384),i * 384:(i * 384 + 384)].sum().item()
                
                
        # gt_cnt = gt_dots.shape[1]
        if isinstance(output,dict):
            pred_cnt = pred_cnt/len(list(output.keys()))
        cnt_err = abs(pred_cnt - gt_cnt)
        if cnt_err > 40:
            print(im_id)
        train_mae += cnt_err
        train_rmse += cnt_err ** 2
        print(
            f'{data_iter_step}/{len(dataloader.dataset)}: pred_cnt: {pred_cnt},  gt_cnt: {gt_cnt},  error: {cnt_err},  AE: {cnt_err},  SE: {cnt_err ** 2} ')

        loss_array.append(cnt_err)
        gt_array.append(gt_cnt)
        pred_array.append(pred_cnt)
        scale_list_x.append(scale[0][0][0].item())
        scale_list_y.append(scale[0][0][1].item())
        scale_list.append(math.sqrt(scale[0][0][0].item() * scale[0][0][1].item() * 384 * 384))
        if vis:

            for i in range(len(output_list)):
                kernel_size = atten_map.shape[-1] + 1 - output_list[i].shape[-1]
                output_list[i] = output_list[i].reshape(-1)
                output_list[i] = output_list[i].unsqueeze(1).repeat(1,kernel_size ** 2)
                output_list[i] = F.fold(output_list[i].T, output_size=(24,24), kernel_size=kernel_size, stride=1).squeeze(0)

            merge_out = torch.hstack(output_list).repeat_interleave(16, dim=0).repeat_interleave(16, dim=1)
            merge_attn = torch.hstack(merge_attn_list) 

            
            k = min(math.floor(int(pred_cnt * (scale[0][:,0].mean() * 16) * (scale[0][:,1].mean() * 16))),merge_attn.reshape(-1).shape[0])
            threshold = torch.topk(merge_attn.reshape(-1), k)[0].min()
            modify_attn = (merge_attn > (threshold - 1e-5)).float()
            modify_attn = modify_attn.repeat_interleave(16, dim=0).repeat_interleave(16, dim=1)
            modify_attn = torch.from_numpy(ndimage.gaussian_filter((modify_attn).detach().cpu().numpy(), sigma=(5, 5), order=0)).to(device)
            merge_out = (merge_out * modify_attn)

            # # case1: detection like
            merge_out = (merge_out > 0).float()

            # # case2: density like
            # merge_out = torch.from_numpy(np.log1p(merge_out.detach().cpu().numpy()))

            # print(merge_out.shape)

            merge_lc = torch.hstack(output_list)
            dh, dw = merge_out.shape[-2:]
            visualize_single_image_local(image[:,:,:,pl:pl+w], merge_out[:,pl:pl+w], gt_density[:, :,pl:pl+w],merge_lc,save_path=os.path.join(vis,im_id[0]),pred_cnt=pred_cnt)
        torch.cuda.synchronize(device=device)

    mae = train_mae / (len(dataloader.dataset))
    mse = (train_rmse / (len(dataloader.dataset))) ** 0.5
    r2 = r2_score(y_pred=pred_array,y_true=gt_array)
    y_pred, y_gt = np.array(pred_array), np.array(gt_array)
    rmae = np.mean(np.abs((y_pred - y_gt)/y_gt))
    rrmse = np.sqrt(np.mean(((y_gt - y_pred) / y_gt) ** 2))
    wca = 1 - abs(y_pred - y_gt).sum() / y_gt.sum()

    print('Current MAE: {:5.2f}, RMSE: {:5.2f}, rMAE: {:5.2f}, rMSE: {:5.2f}, WCA: {:5.2f}, R2: {:5.2f} '.format(mae, mse, rmae, rrmse, wca, r2))
    return mae, mse, wca, r2
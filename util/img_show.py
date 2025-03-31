from torchvision import transforms
from PIL import Image
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
def img_save(img, pth='/data/wangzhicheng/Code/CntViT/test.jpg', flag = 2):
    ### if flag = 0
    ### 用来显示 attention 图 attention map = (K*Q)*V/scale
    if flag == 0:
        n = 1
        c = 512
        h = 24
        w = 24
        # img = img.transpose(1, 2).reshape(n, c, h, w)
        img = rearrange(img,'b (h w) c-> b c h w',h = 24, w = 24)
        img = img[0,0,:,:]
        uploader = transforms.ToPILImage()
        img = img.cpu().clone()
        img = uploader(img)
        img.convert('RGB')
        img.save(pth)
    ### 直接保存第一个batch，第一个channel的图片
    elif flag == 1:
        img1 = img[0,:,:,:].squeeze(0)
        uploader = transforms.ToPILImage()
        img1 = img1.cpu().clone()
        img1 = uploader(img1)
        img1.save(pth)
    elif flag == 2:
        img1 = img
        uploader = transforms.ToPILImage()
        img1 = img1.cpu().clone()
        img1 = uploader(img1)
        img1.save(pth)

def img_save_color(img, pth='/data/wangzhicheng/Code/CntViT/test.jpg',color = 'jet'):
    ### if flag = 0
    ### 用来显示 attention 图 attention map = (K*Q)*V/scale
    uploader = transforms.ToPILImage()
    img1 = img.cpu().clone().numpy()
    # cmap = 'seismic'
    cmap = color ## 密度图用的颜色
    img1 = img1.astype(np.uint8)
    plt.figure(dpi=800)
    plt.imshow(img1, cmap=plt.get_cmap(cmap))
    plt.axis('off')
    plt.savefig(pth,bbox_inches='tight')
    img1 = uploader(img1)
    plt.close()




def picture_fuse(img1,img2,pth='/data/wangzhicheng/Code/CntViT/PaperFig/fix.jpg',alpha=0.5,beta=0.5,color='jet'):
    w = 384
    h = 384

    img10 = img1.squeeze(0)*255
    img10 = rearrange(img10,'c w h->w h c')
    uploader = transforms.ToPILImage()
    img10 = img10.cpu().clone().numpy()
    img10 = img10.astype('uint8')
    
    img2 = img2.cpu().clone().numpy() 
    norm_img2 = np.zeros(img2.shape)
    norm_img2 = cv2.normalize(img2, norm_img2, 0, 255, cv2.NORM_MINMAX)
    norm_img2 = np.asarray(norm_img2, dtype=np.uint8)

    heat_img = cv2.applyColorMap(norm_img2, cv2.COLORMAP_JET) # 注意此处的三通道热力图是cv2专有的GBR排列
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)# 将BGR图像转为RGB图像

    combine = cv2.addWeighted(cv2.resize(img10,(w,h)),alpha,cv2.resize(heat_img,(w,h)),beta,0)
    plt.figure(dpi=800)
    plt.imshow(combine)
    plt.axis('off')
    plt.savefig(pth,bbox_inches='tight')
    plt.close()

# def picture_fuse(img1,img2,pth='/data/wangzhicheng/Code/CntViT/PaperFig/fix.jpg',alpha=0.5,beta=0.5,color='jet'):
#     w = 384
#     h = 384

#     img10 = img1.squeeze(0)*255
#     img10 = rearrange(img10,'c w h->w h c')
#     uploader = transforms.ToPILImage()
#     img10 = img10.cpu().clone().numpy()
#     img10 = img10.astype('uint8')
#     img10 = cv2.resize(img10,(w,h))
#     # img2 = img2.cpu().clone().numpy() 
#     # norm_img2 = np.zeros(img2.shape)
#     # norm_img2 = cv2.normalize(img2, norm_img2, 0, 255, cv2.NORM_MINMAX)
#     # norm_img2 = np.asarray(norm_img2, dtype=np.uint8)

#     # heat_img = cv2.applyColorMap(norm_img2, cv2.COLORMAP_JET) # 注意此处的三通道热力图是cv2专有的GBR排列
#     # heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)# 将BGR图像转为RGB图像
    

#     cmap = plt.cm.get_cmap('jet')
#     density_map = img2
#     density_map = torch.nn.functional.interpolate(density_map.unsqueeze(0).unsqueeze(0), (h, w), mode='bilinear').squeeze(0).squeeze(0).cpu().numpy()
#     #gt_density = torch.nn.functional.interpolate(gt_density, (h, w), mode='bilinear').squeeze(0).squeeze(0).cpu().numpy()

#     density_map = cmap(density_map / (density_map.max()) + 1e-14) * 255.0
#     density_map = density_map[:,:,0:3] * beta + img10 * alpha
#     density_map = density_map.astype(np.uint8)
#     #gt_density = cmap(gt_density / (gt_density.max()) + 1e-14) * 255.0
#     #gt_density = gt_density[:,:,0:3] * 0.5 + img10 * 0.5



#     # combine = cv2.addWeighted(cv2.resize(img10,(w,h)),alpha,cv2.resize(heat_img,(w,h)),beta,0)
#     plt.figure(dpi=800)
#     plt.imshow(density_map)
#     plt.axis('off')
#     plt.savefig(pth,bbox_inches='tight')
#     plt.close()

def concatenate_img(img_list):
    list1 = []
    for i in range(9):
        if img_list[i].shape[0] == 3:
            img10 = img_list[i].squeeze(0) 
            img10 = rearrange(img10,'c w h->w h c')
        else:
            img10 = img_list[i] 
        img10 = img10.cpu().clone().numpy()
        list1.append(img10)
    img_1 = np.concatenate(list1[:3], axis=0)
    img_2 = np.concatenate(list1[3:6], axis=0)
    img_3 = np.concatenate(list1[6:], axis=0)

    img = np.concatenate([img_1,img_2,img_3],axis=1)
    img = img * 255
    img = img.astype(np.uint8)
    return img

def concatenate_density(img_list):
    list1 = []
    for i in range(9):
        if img_list[i].shape[0] == 3:
            img10 = img_list[i].squeeze(0) 
            img10 = rearrange(img10,'c w h->w h c')
        else:
            img10 = img_list[i] 
        img10 = img10.cpu().clone().numpy()
        list1.append(img10)
    img_1 = np.concatenate(list1[:3], axis=0)
    img_2 = np.concatenate(list1[3:6], axis=0)
    img_3 = np.concatenate(list1[6:], axis=0)

    img = np.concatenate([img_1,img_2,img_3],axis=1)
    img = (img/np.max(img)) * 255 
    img = img.astype(np.uint8)
    return img

def picture_fuse_N(img1_list,img2_list,pth='/data/wangzhicheng/Code/CntViT/PaperFig/fix.jpg',alpha=0.5,beta=0.5,color='jet'):
    # list1 = []
    # for i in range(9):
    #     img10 = img1_list[i].squeeze(0)*255
    #     img10 = rearrange(img10,'c w h->w h c')
    #     img10 = img10.cpu().clone().numpy()
    #     list1.append(img10)
    # img_1 = np.concatenate(list1[:3], axis=0)
    # img_2 = np.concatenate(list1[3:6], axis=0)
    # img_3 = np.concatenate(list1[6:], axis=0)

    # img = np.concatenate([img_1,img_2,img_3],axis=1)
    # img = img.astype(np.uint8)
    w = 384
    h = 384
    img1 = concatenate_img(img1_list)
    img1 = cv2.normalize(img1, img1, 0, 255, cv2.NORM_MINMAX)
    img2 = concatenate_density(img2_list)
    

    # b, g, r = cv2.split(img2)
    # img2 = cv2.merge([r, g, b])
    # img2 = img2.cpu().clone().numpy() 
    norm_img2 = np.zeros(img2.shape)
    norm_img2 = cv2.normalize(img2, norm_img2, 0, 255, cv2.NORM_MINMAX)
    norm_img2 = np.asarray(norm_img2, dtype=np.uint8)

    heat_img = cv2.applyColorMap(norm_img2, cv2.COLORMAP_JET) # 注意此处的三通道热力图是cv2专有的GBR排列
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)# 将BGR图像转为RGB图像
    # a = cv2.resize(img10,dsize=(w,h))
    # b = cv2.resize(img2,dsize=(w,h))
    combine = cv2.addWeighted(cv2.resize(img1,(w,h)),alpha,cv2.resize(heat_img,(w,h)),beta,0)
    # cmap = color
    plt.figure(dpi=800)
    plt.imshow(combine)
    plt.axis('off')
    plt.savefig(pth,bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    a = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            a[i,j] = (i*4+j)/16
    a = a * 255
    a = a.astype(np.uint8)
    a = cv2.applyColorMap(a,colormap=cv2.COLORMAP_JET)
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    plt.imshow(a)
    plt.axis('off')
    pth = '/data/wangzhicheng/Code/CntViT/PaperFig/test.jpg'
    plt.savefig(pth,bbox_inches='tight')

 


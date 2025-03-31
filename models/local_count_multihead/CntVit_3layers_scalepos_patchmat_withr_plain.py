from functools import partial
from pathlib import Path
import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed
import sys
import os
if __name__ == '__main__':
    sys.path.append(os.getcwd())
from models.local_count_multihead.Block.Blocks_etop import Block,flash_Block
import torch.nn.functional as F
from util.pos_embed import get_2d_sincos_pos_embed
import numpy as np
from thop import profile
from thop import clever_format
from util.img_show import img_save,img_save_color

class SupervisedMAELoose(nn.Module):
    """ CntVit with VisionTransformer backbone
    """

    def __init__(self, img_size=384, patch_size=16, in_chans=3, 
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, device = 'cuda:0',test = False,
                 norm_pix_loss=False, drop_path_rate = 0.3,
                 interaction_indexes = [[0, 2], [3, 5], [6, 8], [9, 11]],with_cffn=True, cffn_ratio=0.25,use_extra_extractor=True,
                 blocksize_list=[32,64,128], output_stride_list=[16,32,64],
                 mode = 'GlobalAttention' ,decodemode = 'GlobalAttention', similarityfunc = 'PatchConv',similaritymode = 'OutputAdd',gamma = True, xenhance=False,mullayer = True, updown=None):
        super().__init__()
        ## Setting the model
        self.mode = mode
        self.mullayer = mullayer
        self.decodemode = decodemode
        self.similarityfunc = similarityfunc
        self.similaritymode = similaritymode
        self.gamma = gamma
        self.updown = updown
        self.test = test

        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        ## Setting the model

        ## Global Setting
        self.patch_size = patch_size
        self.img_size = img_size
        ex_size = 64
        self.norm_pix_loss = norm_pix_loss
        ## Global Setting

        ## Encoder specifics
        self.scale_embeds = nn.Linear(2, embed_dim, bias=True)
        self.patch_embed_exemplar = PatchEmbed(ex_size, patch_size, in_chans + 1, embed_dim)
        num_patches_exemplar = self.patch_embed_exemplar.num_patches
        self.pos_embed_exemplar = nn.Parameter(torch.zeros(1, num_patches_exemplar, embed_dim), requires_grad=False)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.norm = norm_layer(embed_dim)
        if self.mode == 'GlobalAttention':
            self.blocks = nn.ModuleList([
            flash_Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth-1)])
        self.blocks.append(Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer))
        self.v_y = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True)
        self.density_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim)

        self.accm = {}
        self.counter = nn.ModuleDict()

        for iter, blocksize in enumerate(blocksize_list):
            blocksize, os = blocksize // 16, output_stride_list[iter]//16
            num_patch = self.patch_embed.img_size[0] // 16
            target_size = int((num_patch - blocksize) / os + 1)
            accm_12 = torch.FloatTensor(1, blocksize * blocksize, target_size * target_size).fill_(1)
            accm_12 = F.fold(accm_12, (num_patch, num_patch), kernel_size=blocksize, stride=os)
            accm_12 = 1 / accm_12
            accm_12 /= blocksize ** 2
            self.accm[f'{blocksize * 16}'] = F.unfold(accm_12, kernel_size=blocksize, stride=os).sum(1).view(1, 1, target_size, target_size)
            self.counter[f'{blocksize * 16}'] = nn.Sequential(
                nn.Conv2d(769,512,kernel_size=3,stride=1,padding=1,bias=False),
                nn.GroupNorm(32,512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512,256,kernel_size=3,stride=1,padding=1,bias=False),
                nn.GroupNorm(32,256),
                nn.ReLU(inplace=True),
                nn.AvgPool2d((blocksize, blocksize), stride=os),
                nn.Conv2d(256, 256, kernel_size=1, stride=1),
                nn.GroupNorm(32,256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 1, 1)
            )
            
        self.initialize_weights()
        for name, p in self.named_parameters():
            print(f'{name}')

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        pos_embde_exemplar = get_2d_sincos_pos_embed(self.pos_embed_exemplar.shape[-1],
                                                     int(self.patch_embed_exemplar.num_patches ** .5), cls_token=False)
        self.pos_embed_exemplar.copy_(torch.from_numpy(pos_embde_exemplar).float().unsqueeze(0))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.constant_(m.weight, 0.01)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0.2)
            nn.init.constant_(m.bias, 1)



    def scale_embedding(self, exemplars, scale_infos):
        method = 1
        if method == 0:
            bs, n, c, h, w = exemplars.shape
            scales_batch = []
            for i in range(bs):
                scales = []
                for j in range(n):
                    w_scale = torch.linspace(0, scale_infos[i, j, 0], w)
                    w_scale = repeat(w_scale, 'w->h w', h=h).unsqueeze(0)
                    h_scale = torch.linspace(0, scale_infos[i, j, 1], h)
                    h_scale = repeat(h_scale, 'h->h w', w=w).unsqueeze(0)
                    scale = torch.cat((w_scale, h_scale), dim=0)
                    scales.append(scale)
                scales = torch.stack(scales)
                scales_batch.append(scales)
            scales_batch = torch.stack(scales_batch)

        if method == 1:
            bs, n, c, h, w = exemplars.shape
            scales_batch = []
            for i in range(bs):
                scales = []
                for j in range(n):
                    w_scale = torch.linspace(0, scale_infos[i, j, 0], w)
                    w_scale = repeat(w_scale, 'w->h w', h=h).unsqueeze(0)
                    h_scale = torch.linspace(0, scale_infos[i, j, 1], h)
                    h_scale = repeat(h_scale, 'h->h w', w=w).unsqueeze(0)
                    scale = w_scale + h_scale
                    scales.append(scale)
                scales = torch.stack(scales)
                scales_batch.append(scales)
            scales_batch = torch.stack(scales_batch)

        scales_batch = scales_batch.to(exemplars.device)
        exemplars = torch.cat((exemplars, scales_batch), dim=2)

        return exemplars

    def forward_encoder(self, x, y, scales=None):
        if self.mode == 'GlobalAttention':
            y_embed = []
            y = rearrange(y,'b n c w h->n b c w h')
            for box in y:
                box = self.patch_embed_exemplar(box)
                box = box + self.pos_embed_exemplar
                y_embed.append(box)
            y_embed = torch.stack(y_embed, dim=0)
            box_num,_,n,d = y_embed.shape
            y = rearrange(y_embed, 'box_num batch n d->batch (box_num  n) d')
            x = self.patch_embed(x)
            x = x + self.pos_embed
            _, l, d = x.shape
            attns = []
            x_y = torch.cat((x,y),axis=1)
            for i, blk in enumerate(self.blocks):
                x_y, attn = blk(x_y)
                attns.append(attn)
            x_y = self.norm(x_y)     ## 输出�???? [batch * 288 * 768] 仅仅保存了一�????
            x = x_y[:,:l,:]
            for i in range(box_num):
                y[:,i*n:(i+1)*n,:] = x_y[:,l+i*n:l+(i+1)*n,:]
            y = rearrange(y,'batch  (box_num  n) d->box_num batch n d',box_num = box_num,n=n)
            return x, y, attns

    def AttentionEnhance_for_fenxi(self, attns, l=24, n=1, layer=0, fig_name='0.jpg'):
        output_dir = '/data/wangzhicheng/Code/CntViT/attention/' + fig_name[0] + '/layer_' + str(layer)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        l_x = int(l * l)
        l_y = int(4 * 4)
        r = self.img_size // self.patch_size

        # attns = torch.stack(attns) # 8 * batch * heads * (M+l*3) * (M+l*3)
        attns = torch.mean(attns, dim=1)

        attns_y2y = attns[:, l_x:, l_x:]
        n = 3
        for i in range(n):
            attns_y2yi = attns[:, l_x + i * l_y:l_x + (i + 1) * l_y, l_x + i * l_y:l_x + (i + 1) * l_y]
            attns_y2yi = torch.mean(attns_y2yi, dim=1, keepdim=True)
            attns_y2yi = rearrange(attns_y2yi, 'b l (w h)->b l w h', w=4)
            nameDi = output_dir + '/attnD' + str(i) + '.jpg'
            img_save_color(attns_y2yi[0, 0] * 255 / torch.max(attns_y2yi[0]), pth=nameDi)

        attns_y2x = attns[:, :l_x, l_x:]
        attns_x2y = attns[:, l_x:, :l_x]
        attns_y2x = rearrange(attns_y2x, 'a b c->a c b')
        patch_attn = attns[:, :l_x, :l_x]  ## 576 * 576
        patch_attn = torch.mean(patch_attn, dim=1, keepdim=True)
        patch_attn = rearrange(patch_attn, 'b l (w h)->b l w h', w=r)
        nameA = output_dir + '/attnA.jpg'
        img_save_color(patch_attn[0, 0] * 255 / torch.max(patch_attn[0]), pth=nameA)

        attns_x2y = rearrange(attns_x2y, 'b (n ly) l->b n ly l', ly=l_y)
        attns_y2x = rearrange(attns_y2x, 'b (n ly) l->b n ly l', ly=l_y)
        attns_x2y = attns_x2y.sum(2)
        attns_y2x = attns_y2x.sum(2)

        attns_x2y = torch.mean(attns_x2y, dim=1).unsqueeze(-1)
        attns_y2x = torch.mean(attns_y2x, dim=1).unsqueeze(-1)

        attns_x2y = rearrange(attns_x2y, 'b (w h) c->b c w h', w=r, h=r)
        attns_y2x = rearrange(attns_y2x, 'b (w h) c->b c w h', w=r, h=r)
        nameB = output_dir + '/attnB.jpg'
        nameC = output_dir + '/attnC.jpg'
        img_save_color(attns_x2y[0, 0] * 255 / torch.max(attns_x2y[0]), pth=nameC)
        img_save_color(attns_y2x[0, 0] * 255 / torch.max(attns_y2x[0]), pth=nameB)
        return attns_x2y

    def AttentionEnhance(self, attns, l=24, n=1):
        l_x = int(l * l)
        l_y = int(4 * 4)
        r = self.img_size // self.patch_size

        # attns = torch.stack(attns) # 8 * batch * heads * (M+l*3) * (M+l*3)
        attns = torch.mean(attns, dim=1)

        attns_x2y = attns[:, l_x:, :l_x]
        attns_x2y = rearrange(attns_x2y, 'b (n ly) l->b n ly l', ly=l_y)
        # attns_x2y = rearrange(attns_y2x,'b l (n ly)->b n ly l',ly = l_y)
        attns_x2y = attns_x2y * n.unsqueeze(-1).unsqueeze(-1)
        attns_x2y = attns_x2y.sum(2)

        attns_x2y = torch.mean(attns_x2y, dim=1).unsqueeze(-1)
        attns_x2y = rearrange(attns_x2y, 'b (w h) c->b c w h', w=r, h=r)
        return attns_x2y

    def MacherMode(self, x, y, attn:list, scales=None, name='0.jpg', vis=True):
        if self.similaritymode == 'OutputAdd':
            # x = self.decoder_norm(x)
            B, L, D = x.shape
            # y = self.decoder_norm(y)
            n,B,_,D = y.shape
            r2 = (scales[:, :, 0] + scales[:, :, 1]) ** 2
            n = 16 / (r2 * 384)
            # density_feature = rearrange(x, 'b (w h) d->b d w h', w=24)
            density_feature = rearrange(x, 'b (w h) d->b d w h', w=int(self.patch_embed.img_size[0] // 16))
            if name != None:
                for i in range(12):
                    density_enhance1 = self.AttentionEnhance_for_fenxi(attn[i], l=int(np.sqrt(L)), n=n, layer=i,
                                                                       fig_name=name)
            density_enhance = self.AttentionEnhance(attn[-1], l=int(np.sqrt(L)), n=n)
            # if vis:
            #     temp = density_enhance.squeeze(0).squeeze(0).detach().cpu().numpy()
            #     attention_map = cv2.normalize(temp , None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #     attention_map = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            #     cv2.imwrite("atten_map.png", attention_map)
            density_feature = torch.cat((density_feature, density_enhance), axis=1)
            return density_feature, density_enhance

    def Regressor(self, feature, test, size):
        # feature = self.decode_head(feature)
        f={}
        if not test:
            for iter, (level, counter) in enumerate(self.counter.items()): 
                f.update({level: counter(feature)})
                f[level]  = f[level].squeeze(-3)
        else:
            for iter, (level, counter) in enumerate(self.counter.items()): 
                if size.item() < int(level):
                    break
            # _, _, h, w = f[level].shape()
            f = self.counter[level](feature)
            self.accm[level] = self.accm[level].to(feature.device)
            f *= self.accm[level]

        return f

    def forward(self, samples, size=None, test=True, single=False):  ## 输入的是[8, 3, 384, 384]
        imgs = samples[0]
        boxes = samples[1]
        scales = samples[2]
        size = samples[3]
        if len(samples) > 3:
            name = samples[3][0]
        boxes = self.scale_embedding(boxes, scales)
        latent, y_latent, attns1 = self.forward_encoder(imgs, boxes, scales=scales)
        density_feature, atten_map = self.MacherMode(latent, y_latent, attns1, scales, name=None)
        density_map = self.Regressor(density_feature, test, size)
        if single:
            return density_map, atten_map, attns1
        elif not test:
            return density_map
        else:
            return density_map, atten_map

def local_count_mutihead_loose(**kwargs):
    model = SupervisedMAELoose(
        patch_size=16, embed_dim=768, num_heads=12,
        decoder_embed_dim=512, decoder_depth=3, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),**kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = local_count_mutihead_loose  # decoder: 512 dim, 8 blocks



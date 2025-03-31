from functools import partial
from pathlib import Path
import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed
import sys
sys.path.append('/data/akaxiao/code/hxn/pacvit_update/pacvit_encoder_flash_true_local')
from model.Block.Blocks_etop import Block,flash_Block, FormerBlock,Mlp
import torch.nn.functional as F
from util.pos_embed import get_2d_sincos_pos_embed
import numpy as np
from thop import profile
from thop import clever_format
from util.img_show import img_save,img_save_color


class SupervisedMAE(nn.Module):
    """ CntVit with VisionTransformer backbone
    """

    def __init__(self, img_size=384, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, drop_path_rate=0.3,
                 interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]], with_cffn=True, cffn_ratio=0.25,
                 use_extra_extractor=True,
                 mode='GlobalAttention', decodemode='GlobalAttention', similarityfunc='PatchConv',
                 similaritymode='OutputAdd', gamma=True, xenhance=False, mullayer=True, updown=None):
        super().__init__()
        ## Setting the model
        self.mode = mode
        self.mullayer = mullayer
        self.decodemode = decodemode
        self.similarityfunc = similarityfunc
        self.similaritymode = similaritymode
        self.gamma = gamma
        self.updown = updown

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
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
                for i in range(depth)])
        self.v_y = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True)
        self.density_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed_exemplar = nn.Parameter(torch.zeros(1, num_patches_exemplar, decoder_embed_dim),
                                                       requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_norm = norm_layer(decoder_embed_dim)
        ### decoder blocks
        if self.decodemode == 'GlobalAttention':
            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                      norm_layer=norm_layer)
                for i in range(decoder_depth)])

        ### decoder blocks
        ## Decoder specifics
        ## Regressor
        self.decode_head0 = nn.Sequential(
            nn.Conv2d(513, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        # self.decode_head2 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.GroupNorm(8, 256),
        #     nn.ReLU(inplace=True)
        # )
        # self.decode_head3 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.GroupNorm(8, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 1, kernel_size=1, stride=1)
        # )
        ## Regressor
        self.regressor_12 = nn.Sequential(
            nn.MaxPool2d((3, 3), stride=1, padding=1),
            nn.Conv2d(256, 256, 1, bias=False),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1)
        )
        self.regressor_16 = nn.Sequential(
            nn.MaxPool2d((4, 4), stride=2, padding=1),
            nn.Conv2d(256, 256, 1, bias=False),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1)
        )
        self.regressor_32 = nn.Sequential(
            nn.MaxPool2d((8, 8), stride=4, padding=2),
            nn.Conv2d(256, 256, 1, bias=False),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1)
        )
        kernel_16 = [[1, 1],
                     [1, 1]]
        kernel_16 = torch.FloatTensor(kernel_16).unsqueeze(0).unsqueeze(0)
        self.weight_16 = nn.Parameter(data=kernel_16, requires_grad=False)
        kernel_8 = [[1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1]]
        kernel_8 = torch.FloatTensor(kernel_8).unsqueeze(0).unsqueeze(0)
        self.weight_8 = nn.Parameter(data=kernel_8, requires_grad=False)
        self.get_scale = Mlp(6, 5, 3)

        accm_12 = torch.FloatTensor(1, 3 * 3, 96 * 96).fill_(1).cuda(0)
        accm_12 = F.fold(accm_12, (96, 96), kernel_size=3, stride=1, padding=1)
        accm_12 = 1 / accm_12
        accm_12 /= 3 ** 2
        self.accm_12 = F.unfold(accm_12, kernel_size=3, stride=1, padding=1).sum(1).view(1, 1, 96, 96)

        accm_16 = torch.FloatTensor(1, 4 * 4, 48 * 48).fill_(1).cuda(0)
        accm_16 = F.fold(accm_16, (96, 96), kernel_size=4, stride=2, padding=1)
        accm_16 = 1 / accm_16
        accm_16 /= 4 ** 2
        self.accm_16 = F.unfold(accm_16, kernel_size=4, stride=2, padding=1).sum(1).view(1, 1, 48, 48)

        accm_32 = torch.FloatTensor(1, 8 * 8, 24 * 24).fill_(1).cuda(0)
        accm_32 = F.fold(accm_32, (96, 96), kernel_size=8, stride=4, padding=2)
        accm_32 = 1 / accm_32
        accm_32 /= 8 ** 2
        self.accm_32 = F.unfold(accm_32, kernel_size=8, stride=4, padding=2).sum(1).view(1, 1, 24, 24)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        pos_embde_exemplar = get_2d_sincos_pos_embed(self.pos_embed_exemplar.shape[-1],
                                                     int(self.patch_embed_exemplar.num_patches ** .5), cls_token=False)
        self.pos_embed_exemplar.copy_(torch.from_numpy(pos_embde_exemplar).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        decoder_pos_embed_exemplar = get_2d_sincos_pos_embed(self.decoder_pos_embed_exemplar.shape[-1],
                                                             int(self.patch_embed_exemplar.num_patches ** .5),
                                                             cls_token=False)
        self.decoder_pos_embed_exemplar.data.copy_(torch.from_numpy(decoder_pos_embed_exemplar).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)

        # initialize nn.Linear and nn.LayerNorm
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
            nn.init.constant_(m.weight, 0.1)
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

    def add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def add_level_decode_embed(self, x):
        x[0] = x[0] + self.level_embed_decoder[0]
        x[1] = x[1] + self.level_embed_decoder[1]
        x[2] = x[2] + self.level_embed_decoder[2]
        x[3] = x[3] + self.level_embed_decoder[3]
        return x

    def MulLayerBlock(self):
        stages = []
        scale_factors = [2., 4.]
        out_dim = self.decoder_embed_dim
        out_channels = self.decoder_embed_dim
        dim = self.decoder_embed_dim
        for idx, scale in enumerate(scale_factors):
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    nn.GroupNorm(1, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    # nn.Conv2d(256, 1, kernel_size=1, stride=1)
                    nn.Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=True,
                    ),
                    nn.GroupNorm(1, out_channels),
                    nn.GELU(),
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                    ),
                ]
            )
            layers = nn.Sequential(*layers)
            stages.append(layers)
        self.stages = nn.Sequential(*stages)
        return self.stages

    def forward_encoder(self, x, y, scales=None):
        if self.mode == 'GlobalAttention':
            y_embed = []
            y = rearrange(y, 'b n c w h->n b c w h')
            for box in y:
                box = self.patch_embed_exemplar(box)
                box = box + self.pos_embed_exemplar
                y_embed.append(box)
            y_embed = torch.stack(y_embed, dim=0)
            box_num, _, n, d = y_embed.shape
            y = rearrange(y_embed, 'box_num batch n d->batch (box_num  n) d')
            x = self.patch_embed(x)
            x = x + self.pos_embed
            _, l, d = x.shape
            attns = []
            x_y = torch.cat((x, y), axis=1)
            for i, blk in enumerate(self.blocks):
                x_y, attn = blk(x_y)
                attns.append(attn)
            x_y = self.norm(x_y)  ## 输出为 [batch * 288 * 768] 仅仅保存了一半
            x = x_y[:, :l, :]
            for i in range(box_num):
                y[:, i * n:(i + 1) * n, :] = x_y[:, l + i * n:l + (i + 1) * n, :]
            y = rearrange(y, 'batch  (box_num  n) d->box_num batch n d', box_num=box_num, n=n)
            return x, y, attns

    def forward_decoder(self, x, y, scales=None):
        if self.decodemode == 'GlobalAttention':
            x = self.decoder_embed(x)
            # add pos embed
            x = x + self.decoder_pos_embed
            b, l_x, d = x.shape
            y_embeds = []
            num, batch, l, dim = y.shape
            for i in range(num):
                y_embed = self.decoder_embed(y[i])
                y_embed = y_embed + self.decoder_pos_embed_exemplar
                y_embeds.append(y_embed)
            y_embeds = torch.stack(y_embeds)
            num, batch, l, dim = y_embeds.shape
            y_embeds = rearrange(y_embeds, 'n b l d -> b (n l) d')
            x = torch.cat((x, y_embeds), axis=1)
            attns = []
            xs = []
            ys = []
            for i, blk in enumerate(self.decoder_blocks):
                x, attn = blk(x)
                if i == 2:
                    x = self.decoder_norm(x)
                attns.append(attn)
                xs.append(x[:, :l_x, :])
                ys.append(x[:, l_x:, :])
            return xs, ys, attns

    def to1(self, x, dim=-1):
        value = x.sum(dim).unsqueeze(dim)
        x = x / value
        return x

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

    def MacherMode(self, xs, ys, attn, scales=None, name='0.jpg'):
        if self.similaritymode == 'OutputAdd':
            x = xs[-1]
            # x = self.decoder_norm(x)
            B, L, D = x.shape
            y = ys[-1]
            # y = self.decoder_norm(y)
            B, Ly, D = y.shape
            n = int(Ly / 16)
            r2 = (scales[:, :, 0] + scales[:, :, 1]) ** 2
            n = 16 / (r2 * 384)
            # scales = [1,2,4]
            # density_feature = self.MacherBlock(x,y)
            density_feature = rearrange(x, 'b (w h) d->b d w h', w=24)
            if name != None:
                for i in range(12):
                    density_enhance1 = self.AttentionEnhance_for_fenxi(attn[i], l=int(np.sqrt(L)), n=n, layer=i,
                                                                       fig_name=name)
            density_enhance = self.AttentionEnhance(attn[-1], l=int(np.sqrt(L)), n=n)
            density_feature = torch.cat((density_feature, density_enhance), axis=1)
            return density_feature

    def MacherBlock(self, x, y, n=3):
        ## 对attns处理一下，我们要得到x-y的注意力图，然后得到密度图。
        if self.similarityfunc == 'CrossAttention':
            for i, blk in enumerate(self.CrossBlock):
                result = blk(x, y)
            return result
        elif self.similarityfunc == 'PatchConv' or self.similarityfunc == 'Conv':
            B, L, D = x.shape
            B, Ly, D = y.shape
            ry = int(np.sqrt(Ly / n))
            r = int(np.sqrt(L))
            x = rearrange(x, 'b (h w) d->b d h w', h=r)
            y = rearrange(y, 'b (n l) d->n b d l', n=n)
            y = rearrange(y, 'n b d (h w)->b d n h w', h=ry)

            result, corr = self.matcher(x, y)
            return result

    def Regressor(self, feature, scales):
        feature = F.interpolate(
            self.decode_head0(feature), scale_factor=2, mode='bilinear', align_corners=False)
        feature = F.interpolate(
            self.decode_head1(feature), scale_factor=2, mode='bilinear', align_corners=False)

        # bs=feature.shape[0]
        # h=feature.shape[2]
        # w=feature.shape[3]
        # y_12 = F.unfold(feature.mean(dim=1).view(bs, 1, h, w), kernel_size=3,stride=1,padding=1)
        # p_12 = F.softmax(y_12, dim=1)
        # y_16 = F.unfold(feature.mean(dim=1).view(bs, 1, h, w), kernel_size=4, stride=2, padding=1)
        # p_16 = F.softmax(y_16, dim=1)
        # y_32 = F.unfold(feature.mean(dim=1).view(bs, 1, h, w), kernel_size=8, stride=4, padding=2)
        # p_32 = F.softmax(y_32, dim=1)

        feature_12 = self.regressor_12(feature)
        feature_16 = self.regressor_16(feature)
        feature_32 = self.regressor_32(feature)

        _, _, h, w = feature_12.size()

        feature_12 *= self.accm_12

        feature_16 *= self.accm_16

        feature_32 *= self.accm_32

        # x_12_kxk = feature_12 * p_12.view(bs, 9, feature_12.size()[2], feature_12.size()[3])
        # x_16_kxk = feature_16 * p_16.view(bs, 16, feature_16.size()[2], feature_16.size()[3])
        # x_32_kxk = feature_32 * p_32.view(bs, 64, feature_32.size()[2], feature_32.size()[3])

        feature_12 = F.conv2d(feature_12, self.weight_8, stride=4)
        feature_16 = F.conv2d(feature_16, self.weight_16, stride=2)
        # if self.training:
        #     x_kxk = x_kxk.view(bs, self.k**2, -1)
        #     x_kxk = F.fold(x_kxk, (h,w), kernel_size=self.k)
        #     return x, x_kxk
        # else:
        #     return x_kxk
        # new_scale=self.get_scale(rearrange(scales, 'b n d->b (n d)'))
        # scale_12=new_scale[:,0]
        # scale_12=torch.repeat_interleave(scale_12.unsqueeze(dim=1), repeats=24, dim=1)
        # scale_12 = torch.repeat_interleave(scale_12.unsqueeze(dim=1), repeats=24, dim=1)
        # scale_12 = scale_12.unsqueeze(dim=1)
        #
        # scale_16=new_scale[:,1]
        # scale_16=torch.repeat_interleave(scale_16.unsqueeze(dim=1), repeats=24, dim=1)
        # scale_16 = torch.repeat_interleave(scale_16.unsqueeze(dim=1), repeats=24, dim=1)
        # scale_16 = scale_16.unsqueeze(dim=1)
        #
        # scale_32=new_scale[:,2]
        # scale_32=torch.repeat_interleave(scale_32.unsqueeze(dim=1), repeats=24, dim=1)
        # scale_32 = torch.repeat_interleave(scale_32.unsqueeze(dim=1), repeats=24, dim=1)
        # scale_32 = scale_32.unsqueeze(dim=1)
        #
        # feature_12 *= scale_12
        # feature_16 *= scale_16
        # feature_32 *= scale_32

        feature_12 = feature_12.squeeze(-3)
        feature_16 = feature_16.squeeze(-3)
        feature_32 = feature_32.squeeze(-3)

        f = (feature_12 + feature_16 + feature_32) / 3
        return f

    def forward(self, samples, name=None):  ## 输入的是[8, 3, 384, 384]
        imgs = samples[0]
        boxes = samples[1]
        scales = samples[2]
        if len(samples) > 3:
            name = samples[3][0]
        boxes = self.scale_embedding(boxes, scales)
        latent, y_latent, attns1 = self.forward_encoder(imgs, boxes, scales=scales)
        # latent, y_latent = self.forward_scale(latent, y_latent,scales)
        xs, ys, attns = self.forward_decoder(latent, y_latent)
        density_feature = self.MacherMode(xs, ys, attns, scales, name=None)
        if False:
            density_feature1 = self.MacherMode(xs, ys, attns1, scales, name)
            
        density_map = self.Regressor(density_feature, scales)

        return density_map

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = SupervisedMAE(
        patch_size=16, embed_dim=768, depth=11, num_heads=12,
        decoder_embed_dim=512, decoder_depth=3, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks


if __name__ == '__main__':
    sample = []
    batch = 8
    channel = 3
    size = 384
    num = 3
    img = torch.rand(batch, channel, size, size).cuda(0)
    exemplars = torch.rand(batch, num, channel, 64, 64).cuda(0)
    density = torch.rand(batch, size, size).cuda(0)
    scale = torch.rand(batch, num, 2).cuda(0)
    sample.append(img)
    sample.append(exemplars)
    # sample.append(density)
    sample.append(scale)
    model = mae_vit_base_patch16_dec512d8b().cuda(0)
    output = model(sample)
    flops, params = profile(model, inputs=(sample,))
    flops, params = clever_format([flops, params], '%3.f')
    print(flops)
    print(params)


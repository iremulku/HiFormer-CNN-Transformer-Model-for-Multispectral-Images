import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from einops import rearrange
import torchvision
from timm.models.layers import trunc_normal_
from utils import *
from HiFormer_configs import get_hiformer_s_configs
import ml_collections
import os
import wget


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs('./weights', exist_ok=True)


# HiFormer-S Configs
def get_hiformer_s_configs():
    
    cfg = ml_collections.ConfigDict()

    # Swin Transformer Configs
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 1
    if not os.path.isfile('./weights/swin_tiny_patch4_window7_224.pth'):
        print('Downloading Swin-transformer model ...')
        wget.download("https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth", "./weights/swin_tiny_patch4_window7_224.pth")    
    cfg.swin_pretrained_path = './weights/swin_tiny_patch4_window7_224.pth'

    # CNN Configs
    cfg.cnn_backbone = "resnet34"
    cfg.cnn_pyramid_fm  = [64, 128, 256]
    cfg.resnet_pretrained = True

    # DLF Configs
    cfg.depth = [[1, 1, 0]]
    cfg.num_heads = (3, 3)
    cfg.mlp_ratio=(1., 1., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean((2, 3), keepdim=True)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y.expand_as(x)



class Attention(nn.Module):
    def __init__(self, dim, factor, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim * factor),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, embed_dim=96,
                 depths=[2, 2, 6], num_heads=[3, 6, 12],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, **kwargs):
        
        super().__init__()
        
        patches_resolution = [img_size // patch_size, img_size // patch_size]
        num_patches = patches_resolution[0] * patches_resolution[1]
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio


        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=None)
            self.layers.append(layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


class PyramidFeatures(nn.Module):
    def __init__(self, config, img_size = 224, in_channels=3):
        super().__init__()
        
        model_path = config.swin_pretrained_path
        self.swin_transformer = SwinTransformer(img_size,in_chans = 3)
        checkpoint = torch.load(model_path, map_location=torch.device(device))['model']
        unexpected = ["patch_embed.proj.weight", "patch_embed.proj.bias", "patch_embed.norm.weight", "patch_embed.norm.bias",
                     "head.weight", "head.bias", "layers.0.downsample.norm.weight", "layers.0.downsample.norm.bias",
                     "layers.0.downsample.reduction.weight", "layers.1.downsample.norm.weight", "layers.1.downsample.norm.bias",
                     "layers.1.downsample.reduction.weight", "layers.2.downsample.norm.weight", "layers.2.downsample.norm.bias",
                     "layers.2.downsample.reduction.weight", "norm.weight", "norm.bias"]

        
        resnet = eval(f"torchvision.models.{config.cnn_backbone}(pretrained={config.resnet_pretrained})")
        self.resnet_layers = nn.ModuleList(resnet.children())[:7]
        
        # I add this part new!
        self.se_blocks = nn.ModuleList([
            SEBlock(config.cnn_pyramid_fm[0]),
            SEBlock(config.cnn_pyramid_fm[1]),
            SEBlock(config.cnn_pyramid_fm[2])
        ])
        
        
        
        self.p1_ch = nn.Conv2d(config.cnn_pyramid_fm[0], config.swin_pyramid_fm[0] , kernel_size = 1)
        self.p1_pm = PatchMerging((config.image_size // config.patch_size, config.image_size // config.patch_size), config.swin_pyramid_fm[0])
        self.p1_pm.state_dict()['reduction.weight'][:]= checkpoint["layers.0.downsample.reduction.weight"]
        self.p1_pm.state_dict()['norm.weight'][:]= checkpoint["layers.0.downsample.norm.weight"]
        self.p1_pm.state_dict()['norm.bias'][:]= checkpoint["layers.0.downsample.norm.bias"]        
        self.norm_1 = nn.LayerNorm(config.swin_pyramid_fm[0])
        self.avgpool_1 = nn.AdaptiveAvgPool1d(1) 

        self.p2 = self.resnet_layers[5]
        self.p2_ch = nn.Conv2d(config.cnn_pyramid_fm[1], config.swin_pyramid_fm[1] , kernel_size = 1)
        self.p2_pm = PatchMerging((config.image_size // config.patch_size // 2, config.image_size // config.patch_size // 2), config.swin_pyramid_fm[1])
        self.p2_pm.state_dict()['reduction.weight'][:]= checkpoint["layers.1.downsample.reduction.weight"]
        self.p2_pm.state_dict()['norm.weight'][:]= checkpoint["layers.1.downsample.norm.weight"]
        self.p2_pm.state_dict()['norm.bias'][:]= checkpoint["layers.1.downsample.norm.bias"]           
        
        self.p3 = self.resnet_layers[6]
        self.p3_ch = nn.Conv2d(config.cnn_pyramid_fm[2] , config.swin_pyramid_fm[2] , kernel_size =  1)  
        self.norm_2 = nn.LayerNorm(config.swin_pyramid_fm[2])
        self.avgpool_2 = nn.AdaptiveAvgPool1d(1)    


        for key in list(checkpoint.keys()):
            if key in unexpected or 'layers.3' in key:
                del checkpoint[key]
        self.swin_transformer.load_state_dict(checkpoint)


    def forward(self, x):
        
        for i in range(5):
            x = self.resnet_layers[i](x) 

        # Level 1
        fm1 = x
        ilkb=fm1
        # I add this part new!
        fm1 = self.se_blocks[0](fm1)
        fm1_ch = self.p1_ch(fm1)
        
        fm1_reshaped = Rearrange('b c h w -> b (h w) c')(fm1_ch)               
        sw1 = self.swin_transformer.layers[0](fm1_reshaped)
        sw1_skipped = fm1_reshaped  + sw1
        norm1 = self.norm_1(sw1_skipped) 
        sw1_CLS = self.avgpool_1(norm1.transpose(1, 2))
        sw1_CLS_reshaped = Rearrange('b c 1 -> b 1 c')(sw1_CLS) 
        fm1_sw1 = self.p1_pm(sw1_skipped)
        
        # Level 2
        fm1_sw2 = self.swin_transformer.layers[1](fm1_sw1)
        fm2 = self.p2(fm1)
        # I add this part new!
        # fm2 = self.se_blocks[1](fm2)
        
        fm2_ch = self.p2_ch(fm2)
        fm2_reshaped = Rearrange('b c h w -> b (h w) c')(fm2_ch) 
        fm2_sw2_skipped = fm2_reshaped  + fm1_sw2
        fm2_sw2 = self.p2_pm(fm2_sw2_skipped)
    
        # Level 3
        fm2_sw3 = self.swin_transformer.layers[2](fm2_sw2)
        fm3 = self.p3(fm2)
        ucb=fm3
        # I add this part new!
        # fm3 = self.se_blocks[2](fm3)
        
        fm3_ch = self.p3_ch(fm3)
        fm3_reshaped = Rearrange('b c h w -> b (h w) c')(fm3_ch) 
        fm3_sw3_skipped = fm3_reshaped  + fm2_sw3
        norm2 = self.norm_2(fm3_sw3_skipped) 
        sw3_CLS = self.avgpool_2(norm2.transpose(1, 2))
        sw3_CLS_reshaped = Rearrange('b c 1 -> b 1 c')(sw3_CLS) 

        return [torch.cat((sw1_CLS_reshaped, sw1_skipped), dim=1), torch.cat((sw3_CLS_reshaped, fm3_sw3_skipped), dim=1)], ilkb, ucb

# DLF Module
class All2Cross(nn.Module):
    def __init__(self, config, img_size = 224 , in_chans=3, embed_dim=(96, 384), norm_layer=nn.LayerNorm):
        super().__init__()
        self.cross_pos_embed = config.cross_pos_embed
        self.pyramid = PyramidFeatures(config=config, img_size= img_size, in_channels=in_chans)
        
        n_p1 = (config.image_size // config.patch_size     ) ** 2  # default: 3136 
        n_p2 = (config.image_size // config.patch_size // 4) ** 2  # default: 196 
        num_patches = (n_p1, n_p2)
        self.num_branches = 2
        
        self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])) for i in range(self.num_branches)])
        
        total_depth = sum([sum(x[-2:]) for x in config.depth])
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, total_depth)]  # stochastic depth decay rule
        dpr_ptr = 0
        self.blocks = nn.ModuleList()
        for idx, block_config in enumerate(config.depth):
            curr_depth = max(block_config[:-1]) + block_config[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, num_patches, block_config, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
                                  qkv_bias=config.qkv_bias, qk_scale=config.qk_scale, drop=config.drop_rate, 
                                  attn_drop=config.attn_drop_rate, drop_path=dpr_, norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])

        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=.02)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def forward(self, x):
        xs, ilkb, ucb = self.pyramid(x)

        if self.cross_pos_embed:
          for i in range(self.num_branches):
            xs[i] += self.pos_embed[i]

        for blk in self.blocks:
            xs = blk(xs)
        xs = [self.norm[i](x) for i, x in enumerate(xs)]

        return xs, ilkb, ucb



class ConvUpsample(nn.Module):
    def __init__(self, in_chans=384, out_chans=[128], upsample=True):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        
        self.conv_tower = nn.ModuleList()
        for i, out_ch in enumerate(self.out_chans):
            if i > 0:
                self.in_chans = out_ch
            self.conv_tower.append(nn.Conv2d(
                self.in_chans, out_ch,
                kernel_size=3, stride=1,
                padding=1, bias=False
            ))
            self.conv_tower.append(nn.GroupNorm(32, out_ch))
            self.conv_tower.append(nn.ReLU(inplace=False))
            if upsample:
                self.conv_tower.append(nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False))
            
        self.convs_level = nn.Sequential(*self.conv_tower)
        
    def forward(self, x):
        return self.convs_level(x)


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        super().__init__(conv2d)


class sigmoidOut(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(sigmoidOut, self).__init__()
        self.outsig = nn.Sigmoid()

    def forward(self, x):
        x = self.outsig(x)
        return x  


class HiFormerSE(nn.Module):
    def __init__(self, config=get_hiformer_s_configs(), img_size=224, in_chans=3, n_classes=1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = [4, 16]
        self.n_classes = n_classes
        self.All2Cross = All2Cross(config=config, img_size=img_size, in_chans=in_chans)
        
        self.ConvUp_s = ConvUpsample(in_chans=384, out_chans=[128, 128], upsample=True)
        self.ConvUp_l = ConvUpsample(in_chans=96, upsample=False)
    
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=n_classes,
            kernel_size=3,
        )    

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                128, 16,
                kernel_size=1, stride=1,
                padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)  # Replaces the last Upsample
        )
        self.outsig = sigmoidOut(1, 1)

    
    def forward(self, x):
        xs, ilkb, ucb = self.All2Cross(x)
        embeddings = [x[:, 1:] for x in xs]
        reshaped_embed = []
        for i, embed in enumerate(embeddings):

            embed = Rearrange('b (h w) d -> b d h w', h=(self.img_size // self.patch_size[i]), w=(self.img_size // self.patch_size[i]))(embed)
            embed = self.ConvUp_l(embed) if i == 0 else self.ConvUp_s(embed)
            
            reshaped_embed.append(embed)
               
        # NEW
        conv1x1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1).to(device)
        # NEW
        ilkb=ilkb.to(device)
        ilkbb = conv1x1(ilkb)  # Now it has shape [1, 128, 56, 56]

        # NEW
        con = reshaped_embed[0] + ilkbb

        # MODIFIED
        C = con + reshaped_embed[1]
        # C = reshaped_embed[0] + reshaped_embed[1]
        
        # se_block = SEBlock(in_channels=C.shape[1]).to(device) 
         # C = se_block(C)  # Apply SEBlock
        
        
        C = self.conv_pred(C)

        out = self.segmentation_head(C)
        out = self.outsig(out)
        
        return out  

## PromptIR: Prompting for All-in-One Blind Image Restoration
## Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2306.13090


import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from einops.layers.torch import Rearrange
import time
import numpy as np
import sys
sys.path.append("/mnt/netdisk/liumh/workspace/PromptIR/")
from compress_hyperprior import FeatureCompressor
from torch.nn import CosineSimilarity


# 初始化全局计数器
if 'FILE_INDEX_1' not in globals():
    FILE_INDEX_1 = 1
if 'FILE_INDEX_2' not in globals():
    FILE_INDEX_2 = 1
if 'FILE_INDEX_3' not in globals():
    FILE_INDEX_3 = 1

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    




class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



class resblock(nn.Module):
    def __init__(self, dim):

        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x




##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        

    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt





##########################################################################
##---------- PromptIR -----------------------

class PromptIR(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        decoder = False,
    ):

        super(PromptIR, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        
        self.decoder = decoder
        
        if self.decoder:
            self.prompt1 = PromptGenBlock(prompt_dim=64,prompt_len=5,prompt_size = 64,lin_dim = 96)
            self.prompt2 = PromptGenBlock(prompt_dim=128,prompt_len=5,prompt_size = 32,lin_dim = 192)
            self.prompt3 = PromptGenBlock(prompt_dim=320,prompt_len=5,prompt_size = 16,lin_dim = 384)
        
        
        self.chnl_reduce1 = nn.Conv2d(64,64,kernel_size=1,bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128,128,kernel_size=1,bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320,256,kernel_size=1,bias=bias)



        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64,dim,kernel_size=1,bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2

        self.reduce_noise_channel_2 = nn.Conv2d(int(dim*2**1) + 128,int(dim*2**1),kernel_size=1,bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.reduce_noise_channel_3 = nn.Conv2d(int(dim*2**2) + 256,int(dim*2**2),kernel_size=1,bias=bias)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**2)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1)+192, int(dim*2**2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(dim*2**2) + 512, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim*2**2)+512,int(dim*2**2),kernel_size=1,bias=bias)


        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(dim*2**1) + 224, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim*2**1)+224,int(dim*2**2),kernel_size=1,bias=bias)


        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.noise_level1 = TransformerBlock(dim=int(dim*2**1)+64, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim*2**1)+64,int(dim*2**1),kernel_size=1,bias=bias)


        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
                    
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.save_dir = "/mnt/netdisk/liumh/workspace/PromptIR/dehaze_prompt_origin/" # 存prompt参数

        self.use_compressed_params=False  # 是否使用压缩参数

        self.compressed_param_dir="/mnt/netdisk/liumh/workspace/PromptIR/dehaze_compressed_params"     # 压缩参数目录

        # 如果使用压缩参数，加载压缩模型
        if self.use_compressed_params:
            self.compressor_models = {}
            for param_type in ["dec1", "dec2", "dec3"]:
                # 加载压缩模型
                model_path = f"{self.compressed_param_dir}/models/{param_type}_compressor.pth"
                if param_type == "dec1":
                    compressor = FeatureCompressor(64).cuda()
                elif param_type == "dec2":
                    compressor = FeatureCompressor(128).cuda()
                else:  # dec3
                    compressor = FeatureCompressor(320).cuda()
                
                compressor.load_state_dict(torch.load(model_path))
                compressor.eval()
                self.compressor_models[param_type] = compressor

    def decompress_param(self, param_type, index, device):
        """解压缩参数"""
        compressor = self.compressor_models[param_type]
        
        # 加载压缩数据
        bin_path = f"{self.compressed_param_dir}/{param_type}_compressed/compressed_{index}.bin"
        with open(bin_path, "rb") as f:
            compressed_data = f.read()
        
        # 解压缩
        with torch.no_grad():
            param = compressor.decompress(compressed_data)
        
        return param


    def forward(self, inp_img,noise_emb = None):

        global FILE_INDEX_3,  FILE_INDEX_2,  FILE_INDEX_1  # 明确声明三个 FILE_INDEX 是全局变量

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4)
        if self.decoder:
            if self.use_compressed_params:
                dec3_param = self.decompress_param("dec3", FILE_INDEX_3, inp_img.device)
            else:
                dec3_param = self.prompt3(latent)
                

            # filename = self.save_dir + f"dec3_params/{FILE_INDEX_3}.npy"
            # np.save(filename, dec3_param.cpu().detach().numpy())
            FILE_INDEX_3 += 1
            if dec3_param.shape[2:] != latent.shape[2:]:
                # 如果高度和宽度互换，则调整dec3_param
                if dec3_param.shape[2] == latent.shape[3] and dec3_param.shape[3] == latent.shape[2]:
                    dec3_param = dec3_param.permute(0, 1, 3, 2)  # 交换高度和宽度维度
                # 如果尺寸不匹配，则进行插值调整
                else:
                    dec3_param = F.interpolate(
                        dec3_param, size=latent.shape[2:], 
                        mode='bilinear', align_corners=False
                    )
                    
            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)
                        
        inp_dec_level3 = self.up4_3(latent)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        if self.decoder:
            # if self.use_compressed_params:
            #     dec2_param = self.decompress_param("dec2", FILE_INDEX_2, inp_img.device)
            # else:
            dec2_param = self.prompt2(out_dec_level3)

            # filename = self.save_dir + f"dec2_params/{FILE_INDEX_2}.npy"
            # np.save(filename, dec2_param.cpu().detach().numpy())
            FILE_INDEX_2 += 1
            # print("dec2_param",dec2_param.shape)
            if dec2_param.shape[2:] != out_dec_level3.shape[2:]:
                # 如果高度和宽度互换，则调整dec3_param
                if dec2_param.shape[2] == out_dec_level3.shape[3] and dec2_param.shape[3] == out_dec_level3.shape[2]:
                    dec2_param = dec2_param.permute(0, 1, 3, 2)  # 交换高度和宽度维度
                # 如果尺寸不匹配，则进行插值调整
                else:
                    dec2_param = F.interpolate(
                        dec2_param, size=out_dec_level3.shape[2:], 
                        mode='bilinear', align_corners=False
                    )

            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        if self.decoder:
            # if self.use_compressed_params:
            #     dec1_param = self.decompress_param("dec1", FILE_INDEX_1, inp_img.device)
            # else:
            dec1_param = self.prompt1(out_dec_level2)

            # filename = self.save_dir + f"dec1_params/{FILE_INDEX_1}.npy"
            # np.save(filename, dec1_param.cpu().detach().numpy())
            FILE_INDEX_1 += 1
            # print("dec1_param",dec1_param.shape)
            if dec1_param.shape[2:] != out_dec_level2.shape[2:]:
                # 如果高度和宽度互换，则调整dec3_param
                if dec1_param.shape[2] == out_dec_level2.shape[3] and dec1_param.shape[3] == out_dec_level2.shape[2]:
                    dec1_param = dec1_param.permute(0, 1, 3, 2)  # 交换高度和宽度维度
                # 如果尺寸不匹配，则进行插值调整
                else:
                    dec1_param = F.interpolate(
                        dec1_param, size=out_dec_level2.shape[2:], 
                        mode='bilinear', align_corners=False
                    )

            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)


        out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1








###################################### 250703 ###########################################
# Origin
class PromptIR_Origin(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        decoder = False,
        task_id = 0,
    ):

        super(PromptIR_Origin, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        
        self.decoder = decoder
        
        if self.decoder:
            self.prompt1 = PromptGenBlock(prompt_dim=64,prompt_len=5,prompt_size = 64,lin_dim = 96)
            self.prompt2 = PromptGenBlock(prompt_dim=128,prompt_len=5,prompt_size = 32,lin_dim = 192)
            self.prompt3 = PromptGenBlock(prompt_dim=320,prompt_len=5,prompt_size = 16,lin_dim = 384)
        
        
        self.chnl_reduce1 = nn.Conv2d(64,64,kernel_size=1,bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128,128,kernel_size=1,bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320,256,kernel_size=1,bias=bias)



        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64,dim,kernel_size=1,bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2

        self.reduce_noise_channel_2 = nn.Conv2d(int(dim*2**1) + 128,int(dim*2**1),kernel_size=1,bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3

        self.reduce_noise_channel_3 = nn.Conv2d(int(dim*2**2) + 256,int(dim*2**2),kernel_size=1,bias=bias)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**2)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1)+192, int(dim*2**2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(dim*2**2) + 512, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim*2**2)+512,int(dim*2**2),kernel_size=1,bias=bias)


        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(dim*2**1) + 224, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim*2**1)+224,int(dim*2**2),kernel_size=1,bias=bias)


        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.noise_level1 = TransformerBlock(dim=int(dim*2**1)+64, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim*2**1)+64,int(dim*2**1),kernel_size=1,bias=bias)


        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
                    
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.save_dir = "/mnt/netdisk/liumh/workspace/PromptIR/dehaze_prompt_origin/" # 存prompt参数

        self.use_compressed_params=False  # 是否使用压缩参数

        self.compressed_param_dir="/mnt/netdisk/liumh/workspace/PromptIR/dehaze_compressed_params"     # 压缩参数目录

        self.task_id = task_id 

        # 如果使用压缩参数，加载压缩模型
        if self.use_compressed_params:
            self.compressor_models = {}
            for param_type in ["dec1", "dec2", "dec3"]:
                # 加载压缩模型
                model_path = f"{self.compressed_param_dir}/models/{param_type}_compressor.pth"
                if param_type == "dec1":
                    compressor = FeatureCompressor(64).cuda()
                elif param_type == "dec2":
                    compressor = FeatureCompressor(128).cuda()
                else:  # dec3
                    compressor = FeatureCompressor(320).cuda()
                
                compressor.load_state_dict(torch.load(model_path))
                compressor.eval()
                self.compressor_models[param_type] = compressor

    def decompress_param(self, param_type, index, device):
        """解压缩参数"""
        compressor = self.compressor_models[param_type]
        
        # 加载压缩数据
        bin_path = f"{self.compressed_param_dir}/{param_type}_compressed/compressed_{index}.bin"
        with open(bin_path, "rb") as f:
            compressed_data = f.read()
        
        # 解压缩
        with torch.no_grad():
            param = compressor.decompress(compressed_data)
        
        return param


    def forward(self, inp_img,noise_emb = None):

        global FILE_INDEX_3,  FILE_INDEX_2,  FILE_INDEX_1  # 明确声明三个 FILE_INDEX 是全局变量

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4)
        if self.decoder:
            if self.use_compressed_params:
                dec3_param = self.decompress_param("dec3", FILE_INDEX_3, inp_img.device)
            else:
                dec3_param = self.prompt3(latent)
                

            # filename = self.save_dir + f"dec3_params/{FILE_INDEX_3}.npy"
            # np.save(filename, dec3_param.cpu().detach().numpy())
            FILE_INDEX_3 += 1
            if dec3_param.shape[2:] != latent.shape[2:]:
                # 如果高度和宽度互换，则调整dec3_param
                if dec3_param.shape[2] == latent.shape[3] and dec3_param.shape[3] == latent.shape[2]:
                    dec3_param = dec3_param.permute(0, 1, 3, 2)  # 交换高度和宽度维度
                # 如果尺寸不匹配，则进行插值调整
                else:
                    dec3_param = F.interpolate(
                        dec3_param, size=latent.shape[2:], 
                        mode='bilinear', align_corners=False
                    )
                    
            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)
                        
        inp_dec_level3 = self.up4_3(latent)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        if self.decoder:
            # if self.use_compressed_params:
            #     dec2_param = self.decompress_param("dec2", FILE_INDEX_2, inp_img.device)
            # else:
            dec2_param = self.prompt2(out_dec_level3)

            # filename = self.save_dir + f"dec2_params/{FILE_INDEX_2}.npy"
            # np.save(filename, dec2_param.cpu().detach().numpy())
            FILE_INDEX_2 += 1
            # print("dec2_param",dec2_param.shape)
            if dec2_param.shape[2:] != out_dec_level3.shape[2:]:
                # 如果高度和宽度互换，则调整dec3_param
                if dec2_param.shape[2] == out_dec_level3.shape[3] and dec2_param.shape[3] == out_dec_level3.shape[2]:
                    dec2_param = dec2_param.permute(0, 1, 3, 2)  # 交换高度和宽度维度
                # 如果尺寸不匹配，则进行插值调整
                else:
                    dec2_param = F.interpolate(
                        dec2_param, size=out_dec_level3.shape[2:], 
                        mode='bilinear', align_corners=False
                    )

            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        if self.decoder:
            # if self.use_compressed_params:
            #     dec1_param = self.decompress_param("dec1", FILE_INDEX_1, inp_img.device)
            # else:
            dec1_param = self.prompt1(out_dec_level2)

            # filename = self.save_dir + f"dec1_params/{FILE_INDEX_1}.npy"
            # np.save(filename, dec1_param.cpu().detach().numpy())
            FILE_INDEX_1 += 1
            # print("dec1_param",dec1_param.shape)
            if dec1_param.shape[2:] != out_dec_level2.shape[2:]:
                # 如果高度和宽度互换，则调整dec3_param
                if dec1_param.shape[2] == out_dec_level2.shape[3] and dec1_param.shape[3] == out_dec_level2.shape[2]:
                    dec1_param = dec1_param.permute(0, 1, 3, 2)  # 交换高度和宽度维度
                # 如果尺寸不匹配，则进行插值调整
                else:
                    dec1_param = F.interpolate(
                        dec1_param, size=out_dec_level2.shape[2:], 
                        mode='bilinear', align_corners=False
                    )

            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)


        out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1








# EWC
class PromptIR_EWC(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim=48,
        num_blocks=[4,6,6,8], 
        num_refinement_blocks=4,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        decoder=False,
        task_id=0,
    ):
        super(PromptIR_EWC, self).__init__()
        
        # ===== 核心网络架构 =====
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder = decoder
        
        if self.decoder:
            self.prompt1 = PromptGenBlock(prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96)
            self.prompt2 = PromptGenBlock(prompt_dim=128, prompt_len=5, prompt_size=32, lin_dim=192)
            self.prompt3 = PromptGenBlock(prompt_dim=320, prompt_len=5, prompt_size=16, lin_dim=384)
        
        # 编码器路径
        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64, dim, kernel_size=1, bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim)
        self.reduce_noise_channel_2 = nn.Conv2d(int(dim*2**1) + 128, int(dim*2**1), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1))
        self.reduce_noise_channel_3 = nn.Conv2d(int(dim*2**2) + 256, int(dim*2**2), kernel_size=1, bias=bias)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        # 解码器路径
        self.up4_3 = Upsample(int(dim*2**2))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1)+192, int(dim*2**2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(dim*2**2) + 512, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim*2**2)+512, int(dim*2**2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(dim*2**1) + 224, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim*2**1)+224, int(dim*2**2), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))
        self.noise_level1 = TransformerBlock(dim=int(dim*2**1)+64, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim*2**1)+64, int(dim*2**1), kernel_size=1, bias=bias)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        # 最终输出层
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        # ===== EWC 持续学习策略 =====
        self.task_id = task_id
        self.ewc_params = {}      # 存储旧任务参数
        self.fisher_info = {}     # 存储Fisher信息
        self.ewc_lambda = 1e5    # EWC正则化强度
        
        # 注册梯度钩子
        self._register_ewc_hooks()
    
    def _register_ewc_hooks(self):
        """为关键参数注册梯度钩子实现EWC约束"""
        for name, param in self.named_parameters():
            # 只保护重要的提示参数
            if 'prompt' in name or 'weight' in name:
                param.register_hook(self._create_ewc_hook(name))
    
    def _create_ewc_hook(self, name):
        """创建EWC梯度修改钩子"""
        def hook(grad):
            if self.task_id == 1 and name in self.fisher_info:
                # 应用EWC正则化：λ * F * (θ - θ*)
                ewc_penalty = self.ewc_lambda * self.fisher_info[name] * (self._parameters[name].data - self.ewc_params[name])
                return grad + ewc_penalty
            return grad
        return hook

    def set_ewc_params(self, ewc_params, fisher_info):
        """设置EWC参数（从外部调用）"""
        self.ewc_params = ewc_params
        self.fisher_info = fisher_info

    def forward(self, inp_img, noise_emb=None):
        # 完全保持原始前向传播逻辑
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        
        if self.decoder:
            dec3_param = self.prompt3(latent)
            if dec3_param.shape[2:] != latent.shape[2:]:
                if dec3_param.shape[2] == latent.shape[3] and dec3_param.shape[3] == latent.shape[2]:
                    dec3_param = dec3_param.permute(0, 1, 3, 2)
                else:
                    dec3_param = F.interpolate(
                        dec3_param, size=latent.shape[2:], 
                        mode='bilinear', align_corners=False
                    )
            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)
        
        # ===== 解码器路径 =====
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        
        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3)
            if dec2_param.shape[2:] != out_dec_level3.shape[2:]:
                dec2_param = F.interpolate(dec2_param, size=out_dec_level3.shape[2:], mode='bilinear', align_corners=False)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        if self.decoder:
            dec1_param = self.prompt1(out_dec_level2)
            if dec1_param.shape[2:] != out_dec_level2.shape[2:]:
                dec1_param = F.interpolate(dec1_param, size=out_dec_level2.shape[2:], mode='bilinear', align_corners=False)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)
        
        # ===== 最终输出 =====
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        
        return out_dec_level1



# PIGWM
class PromptIR_PIGWM(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim=48,
        num_blocks=[4,6,6,8], 
        num_refinement_blocks=4,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        decoder=False,
        task_id=0,
    ):
        super(PromptIR_PIGWM, self).__init__()
        
        # ===== 核心网络架构 =====
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder = decoder
        
        if self.decoder:
            self.prompt1 = PromptGenBlock(prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96)
            self.prompt2 = PromptGenBlock(prompt_dim=128, prompt_len=5, prompt_size=32, lin_dim=192)
            self.prompt3 = PromptGenBlock(prompt_dim=320, prompt_len=5, prompt_size=16, lin_dim=384)
        
        # 编码器路径
        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64, dim, kernel_size=1, bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim)
        self.reduce_noise_channel_2 = nn.Conv2d(int(dim*2**1) + 128, int(dim*2**1), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1))
        self.reduce_noise_channel_3 = nn.Conv2d(int(dim*2**2) + 256, int(dim*2**2), kernel_size=1, bias=bias)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        # 解码器路径
        self.up4_3 = Upsample(int(dim*2**2))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1)+192, int(dim*2**2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(dim*2**2) + 512, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim*2**2)+512, int(dim*2**2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(dim*2**1) + 224, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim*2**1)+224, int(dim*2**2), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))
        self.noise_level1 = TransformerBlock(dim=int(dim*2**1)+64, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim*2**1)+64, int(dim*2**1), kernel_size=1, bias=bias)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        # 最终输出层
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        # ===== PIGWM 持续学习策略 =====
        self.task_id = task_id
        self.param_importance = {}      # 存储参数重要性 (I1)
        self.old_params = {}           # 存储旧任务参数 (θ_k^n)
        self.pigwm_lambda = 1e5        # PIGWM正则化强度
        
        # 注册梯度钩子
        self._register_pigwm_hooks()
    
    def _register_pigwm_hooks(self):
        """为关键参数注册PIGWM梯度钩子"""
        for name, param in self.named_parameters():
            param.register_hook(self._create_pigwm_hook(name))
    
    def _create_pigwm_hook(self, name):
        """创建PIGWM梯度修改钩子"""
        def hook(grad):
            if self.task_id == 1 and name in self.param_importance:
                # 计算参数变化量 δθ_k^n = θ_k^{n+1} - θ_k^n
                param_change = self._parameters[name].data - self.old_params[name]
                
                # 计算PIGWM正则化项：I1^T·|δθ| + |δθ|^T·(I1·I1^T)·|δθ|
                importance = self.param_importance[name]
                term1 = torch.sum(importance * torch.abs(param_change))
                term2 = torch.sum((importance * param_change.abs()) @ (importance * param_change.abs()).T)
                
                # 应用PIGWM正则化
                pigwm_penalty = self.pigwm_lambda * (term1 + term2)
                return grad + pigwm_penalty
            return grad
        return hook

    def set_pigwm_params(self, old_params, param_importance):
        """设置PIGWM参数（从外部调用）"""
        self.old_params = old_params
        self.param_importance = param_importance

    def forward(self, inp_img, noise_emb=None):
        # 完全保持原始前向传播逻辑
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        
        if self.decoder:
            dec3_param = self.prompt3(latent)
            if dec3_param.shape[2:] != latent.shape[2:]:
                if dec3_param.shape[2] == latent.shape[3] and dec3_param.shape[3] == latent.shape[2]:
                    dec3_param = dec3_param.permute(0, 1, 3, 2)
                else:
                    dec3_param = F.interpolate(
                        dec3_param, size=latent.shape[2:], 
                        mode='bilinear', align_corners=False
                    )
            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)
        
        # ===== 解码器路径 =====
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        
        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3)
            if dec2_param.shape[2:] != out_dec_level3.shape[2:]:
                dec2_param = F.interpolate(dec2_param, size=out_dec_level3.shape[2:], mode='bilinear', align_corners=False)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        if self.decoder:
            dec1_param = self.prompt1(out_dec_level2)
            if dec1_param.shape[2:] != out_dec_level2.shape[2:]:
                dec1_param = F.interpolate(dec1_param, size=out_dec_level2.shape[2:], mode='bilinear', align_corners=False)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)
        
        # ===== 最终输出 =====
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        
        return out_dec_level1







# DPL
import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptIR_DPL(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim=48,
        num_blocks=[4,6,6,8], 
        num_refinement_blocks=4,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        decoder=False,
        task_id=0,
        M=100,  # 提示池大小
        N=5     # 每个实例选择的提示数
    ):
        super(PromptIR_DPL, self).__init__()
        
        # ===== 核心网络架构 =====
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder = decoder
        
        if self.decoder:
            self.prompt1 = PromptGenBlock(prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96)
            self.prompt2 = PromptGenBlock(prompt_dim=128, prompt_len=5, prompt_size=32, lin_dim=192)
            self.prompt3 = PromptGenBlock(prompt_dim=320, prompt_len=5, prompt_size=16, lin_dim=384)
        
        # 编码器路径
        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64, dim, kernel_size=1, bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim)
        self.reduce_noise_channel_2 = nn.Conv2d(int(dim*2**1) + 128, int(dim*2**1), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1))
        self.reduce_noise_channel_3 = nn.Conv2d(int(dim*2**2) + 256, int(dim*2**2), kernel_size=1, bias=bias)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        # 解码器路径
        self.up4_3 = Upsample(int(dim*2**2))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1)+192, int(dim*2**2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(dim*2**2) + 512, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim*2**2)+512, int(dim*2**2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(dim*2**1) + 224, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim*2**1)+224, int(dim*2**2), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))
        self.noise_level1 = TransformerBlock(dim=int(dim*2**1)+64, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim*2**1)+64, int(dim*2**1), kernel_size=1, bias=bias)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        # 最终输出层
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        # ===== DPL 持续学习策略 =====
        self.task_id = task_id
        self.M = M  # 提示池大小
        self.N = N  # 每个实例选择的提示数
        
        # 图像级提示池
        self.image_prompt_pool = nn.ParameterList([
            nn.Parameter(torch.randn(1, inp_channels, 64, 64))  # 初始提示尺寸64x64
            for _ in range(M)
        ])
        self.image_prompt_freq = torch.ones(M)  # 频率表
        
        # 特征级提示池 - 关键修复：使用自适应通道数
        self.feature_prompt_pool = nn.ParameterList([
            nn.Parameter(torch.randn(1, int(dim*2**3), 8, 8))  # 使用潜在层通道数
            for _ in range(M)
        ])
        self.feature_prompt_freq = torch.ones(M)  # 频率表
        
        # 持续学习参数
        self.param_importance = {}      # 存储参数重要性
        self.old_params = {}             # 存储旧任务参数
        self.pigwm_lambda = 1e5          # 正则化强度
    
    def select_image_prompts(self, x):
        """选择与输入图像最相关的图像级提示"""
        B, C, H, W = x.shape
        
        # 使用统一的嵌入层处理输入图像
        x_embed = self.patch_embed(x)  # [B, dim, H//2, W//2]
        x_embed_pooled = F.adaptive_avg_pool2d(x_embed, (1, 1))  # [B, dim, 1, 1]
        x_embed_pooled = x_embed_pooled.view(B, -1)  # [B, dim]
        
        # 计算每个提示的相似度得分
        scores = []
        for i, prompt in enumerate(self.image_prompt_pool):
            # 调整提示大小以匹配输入分辨率
            resized_prompt = F.interpolate(prompt, size=(H, W), mode='bilinear', align_corners=False)
            prompt_embed = self.patch_embed(resized_prompt)  # [1, dim, H//2, W//2]
            prompt_embed_pooled = F.adaptive_avg_pool2d(prompt_embed, (1, 1))  # [1, dim, 1, 1]
            prompt_embed_pooled = prompt_embed_pooled.view(1, -1)  # [1, dim]
            
            # 计算余弦相似度
            cos_sim = F.cosine_similarity(x_embed_pooled, prompt_embed_pooled.expand(B, -1), dim=1)  # [B]
            
            # 考虑频率的得分
            score = cos_sim * (1 / (self.image_prompt_freq[i] + 1e-8))
            scores.append(score)
        
        scores = torch.stack(scores, dim=1)  # [B, M]
        
        # 获取每个样本前N个最相关的提示索引
        _, topk_indices = torch.topk(scores, self.N, dim=1)  # [B, N]
        
        # 获取相应的提示
        selected_prompts = []
        for i in range(B):
            batch_prompts = []
            for j in range(self.N):
                idx = topk_indices[i, j].item()
                batch_prompts.append(self.image_prompt_pool[idx])
            selected_prompts.append(batch_prompts)
        
        # 更新频率表（仅更新第一个样本的选择）
        for idx in topk_indices[0]:
            self.image_prompt_freq[idx] += 1
        
        return selected_prompts, topk_indices

    def select_feature_prompts(self, feat):
        """选择与输入特征最相关的特征级提示 - 修复维度不匹配问题"""
        B, C, H, W = feat.shape
        
        # 使用自适应池化获取特征表示
        feat_pooled = F.adaptive_avg_pool2d(feat, (1, 1))  # [B, C, 1, 1]
        feat_pooled = feat_pooled.view(B, -1)  # [B, C]
        
        # 计算每个提示的相似度得分
        scores = []
        for i, prompt in enumerate(self.feature_prompt_pool):
            # 调整提示大小以匹配特征图分辨率
            resized_prompt = F.interpolate(prompt, size=(H, W), mode='bilinear', align_corners=False)
            prompt_pooled = F.adaptive_avg_pool2d(resized_prompt, (1, 1))  # [1, C, 1, 1]
            prompt_pooled = prompt_pooled.view(1, -1)  # [1, C]
            
            # 计算余弦相似度
            cos_sim = F.cosine_similarity(feat_pooled, prompt_pooled.expand(B, -1), dim=1)  # [B]
            
            # 考虑频率的得分
            score = cos_sim * (1 / (self.feature_prompt_freq[i] + 1e-8))
            scores.append(score)
        
        scores = torch.stack(scores, dim=1)  # [B, M]
        
        # 获取每个样本前N个最相关的提示索引
        _, topk_indices = torch.topk(scores, self.N, dim=1)  # [B, N]
        
        # 获取相应的提示
        selected_prompts = []
        for i in range(B):
            batch_prompts = []
            for j in range(self.N):
                idx = topk_indices[i, j].item()
                batch_prompts.append(self.feature_prompt_pool[idx])
            selected_prompts.append(batch_prompts)
        
        # 更新频率表（仅更新第一个样本的选择）
        for idx in topk_indices[0]:
            self.feature_prompt_freq[idx] += 1
        
        return selected_prompts, topk_indices

    def forward(self, inp_img):
        B, C, H, W = inp_img.shape
        
        # 1. 选择并应用图像级提示
        img_prompts, img_indices = self.select_image_prompts(inp_img)
        
        # 为每个样本调整选中的提示尺寸并应用到输入图像
        for i in range(B):
            # 获取当前样本的所有提示
            sample_prompts = img_prompts[i]
            
            # 调整每个提示的尺寸并应用
            for prompt in sample_prompts:
                # 调整提示尺寸以匹配输入图像
                resized_prompt = F.interpolate(prompt, size=(H, W), mode='bilinear', align_corners=False)
                
                # 将提示应用到当前样本
                inp_img[i:i+1] = inp_img[i:i+1] + resized_prompt
        
        # 2. 基础网络处理
        out_enc_level1 = self.encoder_level1(self.patch_embed(inp_img))
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        
        # 3. 选择并应用特征级提示
        feature_prompts, feature_indices = self.select_feature_prompts(latent)
        
        # 为每个特征图调整选中的提示尺寸并应用
        for i in range(B):
            # 获取当前样本的所有特征提示
            sample_feature_prompts = feature_prompts[i]
            
            # 调整每个特征提示的尺寸并应用
            for prompt in sample_feature_prompts:
                # 调整提示尺寸以匹配特征图
                resized_prompt = F.interpolate(prompt, size=latent.shape[2:], mode='bilinear', align_corners=False)
                
                # 将提示应用到当前样本的特征图
                # 关键修复：确保通道数匹配
                if resized_prompt.size(1) != latent.size(1):
                    # 使用1x1卷积调整通道数
                    channel_adapter = nn.Conv2d(resized_prompt.size(1), latent.size(1), kernel_size=1).to(latent.device)
                    resized_prompt = channel_adapter(resized_prompt)
                
                latent[i:i+1] = latent[i:i+1] + resized_prompt
        
        # 4. 解码路径
        inp_dec_level3 = self.up4_3(latent)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        # 5. 最终输出
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        
        return out_dec_level1

    def after_train_task(self, task_id):
        """训练任务后更新PIGWM参数"""
        self.task_id = task_id
        
        # 保存当前参数值
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.old_params[name] = param.data.clone()
        
        # 重置重要性参数
        self.param_importance = {}
    
    def calc_pigwm_loss(self):
        """计算PIGWM正则化损失"""
        pigwm_loss = 0.0
        for name, param in self.named_parameters():
            if name in self.param_importance and name in self.old_params:
                importance = self.param_importance[name].to(param.device)
                old_param = self.old_params[name].to(param.device)
                # 计算参数变化的正则化损失
                pigwm_loss += torch.sum(importance * (param - old_param)**2)
        return self.pigwm_lambda * pigwm_loss




# EcoDPL

class PromptIR_EcoDPL(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim=48,
        num_blocks=[4,6,6,8], 
        num_refinement_blocks=4,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        task_id=0,
        num_components=100,  # 提示组件数量
        num_selected=5       # 每个实例选择的提示数
    ):
        super(PromptIR_EcoDPL, self).__init__()
        
        # ===== 基础网络架构 =====
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        # 编码器路径
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1))
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        # 解码器路径
        # 关键修复：修正上采样层输入通道数
        self.up4_3 = Upsample(int(dim*2**3))  # 输入通道应为384
        
        # 关键修复：修正通道数不匹配问题
        self.reduce_chan_level3 = nn.Conv2d(2 * int(dim*2**2), int(dim*2**2), kernel_size=1, bias=bias)
        
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        # 关键修复：修正上采样层输入通道数
        self.up3_2 = Upsample(int(dim*2**2))  # 输入通道应为192
        
        # 关键修复：修正通道数不匹配问题
        self.reduce_chan_level2 = nn.Conv2d(2 * int(dim*2**1), int(dim*2**1), kernel_size=1, bias=bias)
        
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        # 关键修复：修正上采样层输入通道数
        self.up2_1 = Upsample(int(dim*2**1))  # 输入通道应为96
        
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        # 最终输出层
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        # ===== CODA-Prompt 持续学习策略 =====
        self.task_id = task_id
        self.num_components = num_components
        self.num_selected = num_selected
        
        # 三个提示位置的组件（P, K, A）
        self.prompt_components = nn.ModuleDict({
            "latent": self._init_prompt_components(int(dim*2**3)),
            "level3": self._init_prompt_components(int(dim*2**2)),
            "level2": self._init_prompt_components(int(dim*2**1))
        })
        
        # 当前激活的组件范围（用于任务增量学习）
        self.active_components = {
            "latent": range(0, num_components),
            "level3": range(0, num_components),
            "level2": range(0, num_components)
        }
        
        # 正交正则化强度
        self.ortho_lambda = 0.1
        
        # 持续学习参数
        self.old_params = {}
        self.param_importance = {}
        
    def _init_prompt_components(self, dim_size):
        """初始化提示组件（P, K, A）"""
        return nn.ModuleDict({
            "P": nn.ParameterList([nn.Parameter(torch.randn(1, dim_size, 1, 1)) for _ in range(self.num_components)]),
            "K": nn.ParameterList([nn.Parameter(torch.randn(1, dim_size)) for _ in range(self.num_components)]),
            "A": nn.ParameterList([nn.Parameter(torch.randn(1, dim_size)) for _ in range(self.num_components)])
        })
    
    def generate_prompt(self, feature, position):
        """
        生成分解的注意力提示
        参数:
            feature: 输入特征 [B, C, H, W]
            position: 提示位置 ('latent', 'level3', 'level2')
        返回:
            prompt: 生成的提示 [B, C, H, W]
        """
        B, C, H, W = feature.shape
        
        # 1. 计算查询向量
        query = F.adaptive_avg_pool2d(feature, (1, 1)).view(B, -1)  # [B, C]
        
        # 2. 计算注意力权重
        weights = []
        for i in self.active_components[position]:
            # 获取组件参数
            K_i = self.prompt_components[position]["K"][i]  # [1, C]
            A_i = self.prompt_components[position]["A"][i]  # [1, C]
            
            # 应用注意力机制
            attended_query = query * A_i.expand(B, -1)  # [B, C]
            
            # 计算余弦相似度
            cos_sim = F.cosine_similarity(attended_query, K_i.expand(B, -1), dim=1)  # [B]
            weights.append(cos_sim)
        
        weights = torch.stack(weights, dim=1)  # [B, num_active_components]
        
        # 3. 选择top-k组件
        topk_weights, topk_indices = torch.topk(weights, self.num_selected, dim=1)  # [B, num_selected]
        topk_weights = F.softmax(topk_weights, dim=1)
        
        # 4. 生成提示
        prompt = torch.zeros(B, C, H, W, device=feature.device)
        for b in range(B):
            for j in range(self.num_selected):
                comp_idx = topk_indices[b, j]
                P_comp = self.prompt_components[position]["P"][comp_idx]  # [1, C, 1, 1]
                
                # 调整组件尺寸
                resized_comp = F.interpolate(P_comp, size=(H, W), mode='bilinear', align_corners=False)
                prompt[b] += topk_weights[b, j] * resized_comp.squeeze(0)
        
        return prompt
    
    def orthogonality_loss(self):
        """计算正交正则化损失"""
        loss = 0
        for position in self.prompt_components:
            # 获取当前激活组件的参数
            active_P = [self.prompt_components[position]["P"][i] for i in self.active_components[position]]
            active_K = [self.prompt_components[position]["K"][i] for i in self.active_components[position]]
            active_A = [self.prompt_components[position]["A"][i] for i in self.active_components[position]]
            
            if active_P:
                # 堆叠参数
                P_stack = torch.stack([p.squeeze() for p in active_P])  # [num_active, C]
                K_stack = torch.stack([k.squeeze() for k in active_K])  # [num_active, C]
                A_stack = torch.stack([a.squeeze() for a in active_A])  # [num_active, C]
                
                # 计算正交损失
                loss += torch.norm(P_stack @ P_stack.t() - torch.eye(len(active_P), device=P_stack.device))
                loss += torch.norm(K_stack @ K_stack.t() - torch.eye(len(active_K), device=K_stack.device))
                loss += torch.norm(A_stack @ A_stack.t() - torch.eye(len(active_A), device=A_stack.device))
        
        return self.ortho_lambda * loss
    
    def after_train_task(self, task_id):
        """训练任务后更新组件和参数"""
        self.task_id = task_id
        
        # 保存当前参数值
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.old_params[name] = param.data.clone()
        
        # 重置重要性参数
        self.param_importance = {}
    
    def calc_pigwm_loss(self):
        """计算PIGWM正则化损失"""
        pigwm_loss = 0.0
        for name, param in self.named_parameters():
            if name in self.param_importance and name in self.old_params:
                importance = self.param_importance[name].to(param.device)
                old_param = self.old_params[name].to(param.device)
                # 计算参数变化的正则化损失
                pigwm_loss += torch.sum(importance * (param - old_param)**2)
        return pigwm_loss

    def forward(self, inp_img, noise_emb=None):
        # 编码器路径
        out_enc_level1 = self.encoder_level1(self.patch_embed(inp_img))
        out_enc_level2 = self.encoder_level2(self.down1_2(out_enc_level1))
        out_enc_level3 = self.encoder_level3(self.down2_3(out_enc_level2))
        latent = self.latent(self.down3_4(out_enc_level3))
        
        # 应用latent层提示
        latent_prompt = self.generate_prompt(latent, "latent")
        latent = latent + latent_prompt
        
        # 解码器路径
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        
        # 应用level3提示
        level3_prompt = self.generate_prompt(out_dec_level3, "level3")
        out_dec_level3 = out_dec_level3 + level3_prompt
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        # 应用level2提示
        level2_prompt = self.generate_prompt(out_dec_level2, "level2")
        out_dec_level2 = out_dec_level2 + level2_prompt
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        # 最终输出
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        
        return out_dec_level1
#Implémentation inspirée de nn.labml.ai

from typing import Optional, Tuple, Union, List
import torch
import math
from torch import nn


# fonction d'activation : Swish (x*sigmoid) ou SiLU 
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# classe qui permet de générer un embedding de t
class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.activation = Swish()
        self.n_channels = n_channels
        self.linear1 = nn.Linear(self.n_channels//4, self.n_channels)   # n//4 en entrée car on prendra un vecteur d'embedding de t générer par un sinusoidal encoding
        self.linear2 = nn.Linear(self.n_channels, self.n_channels)
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.n_channels // 8      #half dim par rapport à l'entrée du MLP
        facteur_exp = math.log(10_000) / (half_dim - 1)   #10000 est une constante choisie arbitrairement
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -facteur_exp)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        
        #passage dans le MLP
        emb = self.linear1(emb)
        emb = self.activation(emb)
        emb = self.linear2(emb)
        return emb

# classe qui implémente les blocs résiduels
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int,n_groups: int = 32, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(n_groups, in_channels)    #groupe normalisation plutôt que batch normalisation pour ne pas dépendre de la taille du batch
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()
            
        #embedding de t
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.conv1(self.act1(self.norm1(x)))       #first convolution
        h += self.time_emb(self.time_act(t))[:, :, None, None] #ajout du time embedding
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))     #second convolution
        return h + self.shortcut(x)

# classe qui implémente le bloc d'attention
class AttentionBlock(nn.Module):
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        super().__init__()
        self.n_heads = n_heads
        if d_k is None:
            d_k = n_channels
        self.norm = nn.GroupNorm(n_groups, n_channels)          #normalization layer
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)    #projections for query, key and values 
        self.output = nn.Linear(n_heads * d_k, n_channels)    #final transformation
        self.scale = d_k ** -0.5
        self.d_k = d_k
        
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        _ = t #t n'est pas utilisé ici
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)  #reshape to [batch_size, seq, n_heads * d_k]]
        res = self.output(res)          #transform to [batch_size, seq, n_channels]
        res += x                  #skip connection
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res
    
# première partie du UNet (descente)
class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.res(x, t)
        h = self.attn(h)
        return h

# deuxième partie du UNet (montée)
class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.res(x, t)
        h = self.attn(h)
        return h


# bloc le plus profond
class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.res1(x, t)
        h = self.attn(h)
        h = self.res2(h, t)
        return h
    

class Upsample(nn.Module):            #Scale up the feature map by 2×
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        h = self.conv(x)
        return h
    
class Downsample(nn.Module):        #Scale down the feature map by 2​×
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        h = self.conv(x)
        return h
        
    
class UNet(nn.Module):
    def __init__(
            self, 
            image_channels: int = 3, 
            n_channels: int = 64, 
            ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
            is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, True, True),
            n_blocks: int = 2
        ):
        super().__init__()
        n_resolutions = len(ch_mults)
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))  #Project image into feature map 
        self.time_emb = TimeEmbedding(n_channels * 4)
        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for j in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels*4, is_attn[i]))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))
        self.down = nn.ModuleList(down)
        self.middle = MiddleBlock(out_channels, n_channels * 4)
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for j in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels*4, is_attn[i]))
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels*4, is_attn[i]))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))
        self.up = nn.ModuleList(up)
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) :
        t = self.time_emb(t)
        x = self.image_proj(x)
        h = [x]
        
        for m in self.down:        #première partie du UNet 
            x = m(x,t)
            h.append(x)
            
        x = self.middle(x, t)       #bottom of the UNet
        
        for m in self.up:            #deuxième partie du UNet
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)
        return self.final(self.act(self.norm(x)))  
        
        
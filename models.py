import math, torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=4, qkv_bias = True, dropout = 0.0, fp32_attention = True, qk_norm = True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.qkv_bias = qkv_bias
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.fp32_attention = fp32_attention

        self.qk_norm = qk_norm
        if qk_norm:
            self.ln_q = nn.LayerNorm(dim, eps = 1e-6, elementwise_affine=False, bias=False)
            self.ln_k =  nn.LayerNorm(dim, eps = 1e-6, elementwise_affine=False, bias=False)

    def forward(self, x):
        # Batch Size, N, Dimensionality
        B,N,D = x.shape
        H = self.heads
        d = D // H # Head Size

        q,k,v = self.qkv(x).chunk(3, dim=-1)

        if self.qk_norm:
            q = q.view(B,N,H,d)
            k = k.view(B,N,H,d)
            q = self.ln_q(q.view(B, N, H * d)).view(B, N, H, d).transpose(1, 2)
            k = self.ln_k(k.view(B, N, H * d)).view(B, N, H, d).transpose(1, 2)
        else:
            q = q.view(B,N,H,d).transpose(1, 2) # Batch size, number of heads, N, Head Size
            k = k.view(B,N,H,d).transpose(1, 2)

        v = v.view(B,N,H,d).transpose(1, 2)

        if self.fp32_attention:
            q, k = q.float(), k.float()
        with torch.cuda.amp.autocast(enabled=not self.fp32_attention):
            if self.flash:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
                                                                    attn_mask=None, dropout_p=self.dropout if self.training else 0, 
                                                                    is_causal=False)

            else:
                attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                attn = F.softmax(attn, dim=-1)
                attn = self.dropout(attn)
                y = attn @ v 
        
        y = y.transpose(1, 2).contiguous().view(B, N, D)
        y = self.proj(y)
        return y



class CrossAttention(nn.Module):
    def __init__(self, dim, heads=4, qkv_bias = True, dropout = 0.0, hidden_dim = None, fp32_attention = True, qk_norm = True):
        super().__init__()
        self.hidden_dim = hidden_dim if hidden_dim != None else dim

        self.dim = dim
        self.heads = heads
        self.qkv_bias = qkv_bias
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.q = nn.Linear(dim, self.hidden_dim)
        self.kv = nn.Linear(dim, self.hidden_dim*2)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.fp32_attention = fp32_attention

        self.qk_norm = qk_norm
        if qk_norm:
            self.ln_q = nn.LayerNorm(dim, eps = 1e-6, elementwise_affine=False, bias=False)
            self.ln_k =  nn.LayerNorm(dim, eps = 1e-6, elementwise_affine=False, bias=False)

    def forward(self, x, c):
        # Batch Size, N, Dimensionality
        B,N,D = x.shape
        H = self.heads
        d = self.hidden_dim // H # Head Size

        q = self.q(x).reshape(B, N, H, d)
        k,v = self.kv(c).chunk(2, dim=-1)

        if self.qk_norm:
            q = q.view(B,N,H,d)
            k = k.view(B,N,H,d)
            q = self.ln_q(q.view(B, N, H * d)).view(B, N, H, d).transpose(1, 2)
            k = self.ln_k(k.view(B, N, H * d)).view(B, N, H, d).transpose(1, 2)
        else:
            q = q.view(B,N,H,d).transpose(1, 2) # Batch size, number of heads, N, Head Size
            k = k.view(B,N,H,d).transpose(1, 2)

        v = v.view(B,N,H,d).transpose(1, 2)

        if self.fp32_attention:
            q, k = q.float(), k.float()
        with torch.cuda.amp.autocast(enabled=not self.fp32_attention):
            if self.flash:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
                                                                    attn_mask=None, dropout_p=self.dropout if self.training else 0, 
                                                                    is_causal=False)

            else:
                attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                attn = F.softmax(attn, dim=-1)
                attn = self.dropout(attn)
                y = attn @ v 
        
        y = y.transpose(1, 2).contiguous().view(B, N, D)
        y = self.proj(y)
        return y

        
class FFN(nn.Module):
    def __init__(self, dim, mlp_ratio = 4, activation = "siglu", norm = None, bias=True):
        super().__init__()

        self.activation = activation

        if activation.endswith("glu"):
            hidden_dim = int(2 * (dim * mlp_ratio) / 3)
            self.gate = nn.Linear(dim, hidden_dim, bias = bias) # w1
            self.down = nn.Linear(hidden_dim, dim, bias = bias) # w3
            self.up = nn.Linear(dim, hidden_dim, bias = bias) # w2
        else:
            self.c_fc = nn.Linear(dim, mlp_ratio*dim, bias = bias)
            self.norm = norm if norm is not None else nn.Identity()
            self.c_proj = nn.Linear(mlp_ratio*dim, dim, bias = bias)


        if activation == "gelu" or activation == "geglu":
            self.act = nn.GELU()
        elif activation == "geluaprox":
            self.act = nn.GELU(approximate="tanh")
        elif activation == "silu" or activation == "siglu":
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU() # Default to Relu?
    
    def forward(self, x):
        if self.activation.endswith("glu"):
            return self.down(self.act(self.gate(x)) * self.up(x))
        else:
            x = self.c_fc(x)
            x = self.act(x)
            x = self.norm(x)
            x = self.c_proj(x)
            return x
        
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, dim, heads, ffn_type = "ffn", fp32_attention = True):
        self.norm1 = nn.LayerNorm(dim, elementwise_affine = False) # PixArt uses elementwise_affine False but micro-DiT doesnt?
        self.attn = SelfAttention(dim, heads=heads, fp32_attention=fp32_attention)

        self.cross_attn = CrossAttention(dim, fp32_attention=fp32_attention)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine = False)
        
        self.norm3 = nn.LayerNorm(dim, elementwise_affine = False)

        self.ffn_type = ffn_type
        if ffn_type == "ffn":
            self.ff = FFN(dim)
        else:
            raise "AAAA"
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), # Original DiT uses SiLu, micro-Dit Silu?
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, x, y, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1))
        
        short_residual = x
        normalized = self.norm1(x)
        modulated = modulate(normalized, shift_msa, scale_msa)
        x = self.attn(modulated)
        x = gate_msa.unsqueeze(1) * x
        x += short_residual # PixArt uses DropPath, may be interesting to try

        x = x + self.cross_attn(self.norm2(x), y) # PixArt doesn't use this norm

        short_residual = x
        normalized = self.norm3(x)
        modulated = modulate(normalized, shift_mlp, scale_mlp)
        x = gate_mlp.unsqueeze(1) * self.ff(modulated)
        x += short_residual

        return x
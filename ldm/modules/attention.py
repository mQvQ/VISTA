from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = torch.einsum("bij,bjk->bik", q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        h_ = self.proj_out(h_)

        return x + h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        # x = x.half()
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, image_qkv, st_qkv):
        """
        Apply QKV attention.
        : attention_index_v:[V_len x H] V_len: f*h*w
        : attention_index_a:[A_len, H]
        :param qkv: an [ N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        st_q 对image做attention; 反之同理
        """
        bs, width, img_len = image_qkv.shape
        _, _, st_len = st_qkv.shape

        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        img_q, img_k, img_v = image_qkv.chunk(3, dim=1)
        st_q, st_k, st_v = st_qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))

        w_i = torch.einsum(
            "bct,bcs->bts",
            (img_q * scale).view(bs * self.n_heads, ch, -1),
            (st_k * scale).view(bs * self.n_heads, ch, -1),
        )  # More stable with f16 than dividing afterwards

        w_i = torch.softmax(w_i.float(), dim=-1).type(w_i.dtype)  # [bsz, 1, k_len]
        attn_i = torch.einsum("bts,bcs->bct", w_i, st_v.contiguous().view(bs * self.n_heads, ch, -1)).reshape(
            bs, -1, img_len)

        w_st = torch.einsum(
            "bct,bcs->bts",
            (st_q * scale).view(bs * self.n_heads, ch, -1),
            (img_k * scale).view(bs * self.n_heads, ch, -1),
        )  # More stable with f16 than dividing afterwards
        attn_st = torch.einsum("bts,bcs->bct", w_st, img_v.contiguous().view(bs * self.n_heads, ch, -1)).reshape(
            bs, -1, st_len)

        return attn_i, attn_st
    #
    # @staticmethod
    # def count_flops(model, _x, y):
    #     return count_flops_attn(model, _x, y)

class JointAttention(nn.Module):
    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
    ):
        super().__init__()
        self.channels = channels

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    self.channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"

            self.num_heads = self.channels // num_head_channels

        self.x_norm = normalization(self.channels)
        self.jc_norm = normalization(self.channels)
        self.x_qkv = conv_nd(1, self.channels, self.channels * 3, 1, padding='same')
        self.jc_qkv = conv_nd(1, self.channels, self.channels * 3, 1, padding='same')
        self.attention = QKVAttention(self.num_heads)

        self.x_proj_out = zero_module(conv_nd(1, self.channels, self.channels, 3, padding='same'))


    def forward(self, image, st):

        return checkpoint(self._forward, (image, st), self.parameters(), False)

    def _forward(self, x, joint_context):
        x_token = x = x.permute(0, 2, 1) # dim 1: c
        joint_context_token = joint_context.permute(0, 2, 1)
        x_qkv = self.x_qkv(self.x_norm(x_token))  # [bsz, c, h*w+l]
        jc_qkv = self.jc_qkv(self.jc_norm(joint_context_token))  # [bsz, c, h*w+l]

        x_h, _ = self.attention(x_qkv, jc_qkv)
        x_h = self.x_proj_out(x_h)
        x_h = x + x_h

        return x_h

class MultiTransformerBlock(nn.Module):
    '''
    attn1: self-attn
    attn2: context-1(joint generation modality)
    attn3: context-2(text)
    '''
    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True, checkpoint=False):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = JointAttention(
            channels=dim, num_heads=n_heads) # attention for joint generation modality
        self.attn3 = CrossAttention(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is self-attn if context is none
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.checkpoint = checkpoint

    def forward(self, x, joint_context, context=None):
        return checkpoint(self._forward, (x, joint_context, context), self.parameters(), self.checkpoint)

    def _forward(self, x, joint_context, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(x, joint_context).permute(0, 2, 1)
        x = self.attn3(self.norm2(x), context=context) + x

        x = self.ff(self.norm3(x)) + x
        return x




class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, checkpoint=False)
                for d in range(depth)
            ]
        )
        self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


class STSpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.st_proj_in = nn.Conv1d(in_channels, self.inner_dim, kernel_size=1, stride=1, padding=0)
        self.st_transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(self.inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)
            ]
        )
        self.st_proj_out = zero_module(nn.Conv1d(self.inner_dim, in_channels, kernel_size=1, stride=1, padding=0 ))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, t = x.shape
        s_in = x
        st = self.st_proj_in(self.norm(x))
        st = rearrange(st, "b c t -> b t c")
        for st_block in self.st_transformer_blocks:
            st = st_block(st, context=context)

        st = rearrange(st, "b t c -> b c t")
        st = self.st_proj_out(st)
        return st+s_in

class JointSpatialTransformer(nn.Module):
    """
    Transformer block for image-st data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    # spatial transformer for st & he
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.image_norm = Normalize(in_channels)
        self.st_norm = Normalize(in_channels)

        self.image_proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.st_proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.image_transformer_blocks = nn.ModuleList(
            [
                MultiTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, checkpoint=False)
                for d in range(depth)
            ]
        )

        self.st_transformer_blocks = nn.ModuleList(
            [
                MultiTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, checkpoint=False)
                for d in range(depth)
            ]
        )

        self.image_proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        self.st_proj_out = zero_module(nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0 ))

    def forward(self, image, st, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = image.shape
        i_in, s_in = image, st
        image = self.image_proj_in(self.image_norm(image))
        st = self.st_proj_in(self.st_norm(st))

        for img_block, st_block in zip(self.image_transformer_blocks, self.st_transformer_blocks):
            image = rearrange(image, "b c h w -> b (h w) c")
            st = rearrange(st, "b c t -> b t c")
            image = img_block(image, st, context=context)
            st = st_block(st, image, context=context)


        image = rearrange(image, "b (h w) c -> b c h w", h=h, w=w)
        st = rearrange(st, "b t c -> b c t")
        image = self.image_proj_out(image)
        st = self.st_proj_out(st)
        return image+i_in, st+s_in

class FusionLayer(nn.Module):

    def __init__(self, image_dim, st_dim, dim_out=None):
        super().__init__()
        dim = image_dim**2 + st_dim
        dim_out = dim // 4
        self.norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dim_out=dim_out, mult=4, glu=True, dropout=0.0)

    def forward(self, image, st):
        # 在token 纬度而不是channel进行fusion
        x = torch.cat([image, st], dim=1) # b, t, c
        x = rearrange(x, "b t c  -> b c t")
        x = self.ff(self.norm(x))
        x = rearrange(x, "b c t  -> b t c")
        return x




class JointSpatialTransformerv2(nn.Module):
    """
    Transformer block for image-st data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    不是直接用st和image做context, 做一个混合的adapter 生成st-context 和 image-context
    """
    # spatial transformer for st & he
    def __init__(self, in_channels, image_in_dim, st_in_dim, n_heads, d_head, depth=1, dropout=0.0, context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.image_norm = Normalize(in_channels)
        self.st_norm = Normalize(in_channels)
        # token num
        self.image_in_dim = image_in_dim
        self.st_in_dim = st_in_dim
        self.image_proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.st_proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.image_transformer_blocks = nn.ModuleList(
            [
                MultiTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, checkpoint=False)
                for d in range(depth)
            ]
        )

        self.st_transformer_blocks = nn.ModuleList(
            [
                MultiTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, checkpoint=False)
                for d in range(depth)
            ]
        )
        # fusion layer for joint context
        self.fusion_blocks = nn.ModuleList([
            FusionLayer(image_in_dim, st_in_dim)
            for d in range(depth)
        ])

        self.image_proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        self.st_proj_out = zero_module(nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0 ))

    def forward(self, image, st, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = image.shape
        i_in, s_in = image, st
        image = self.image_proj_in(self.image_norm(image))
        st = self.st_proj_in(self.st_norm(st))

        for img_block, st_block, fusion_layer in zip(self.image_transformer_blocks, self.st_transformer_blocks, self.fusion_blocks):
            image = rearrange(image, "b c h w -> b (h w) c")
            st = rearrange(st, "b c t -> b t c")
            joint_context = fusion_layer(image, st)
            image = img_block(image, joint_context, context=context)
            st = st_block(st, joint_context, context=context)


        image = rearrange(image, "b (h w) c -> b c h w", h=h, w=w)
        st = rearrange(st, "b t c -> b c t")
        image = self.image_proj_out(image)
        st = self.st_proj_out(st)
        return image+i_in, st+s_in

class SplitJointTransformer(nn.Module):

    def __init__(self, ch, num_heads, dim_head, depth, context_dim):
        super().__init__()
        self.image_spatial_transformer = SpatialTransformer(
            ch, num_heads, dim_head, depth, context_dim=context_dim)
        self.st_spatial_transformer = STSpatialTransformer(
            ch, num_heads, dim_head, depth, context_dim=context_dim)

    def forward(self, image, st, context=None):
        image = self.image_spatial_transformer(image, context=context)
        st = self.st_spatial_transformer(st, context=context)

        return image, st

class DualSplitJointTransformer(nn.Module):

    def __init__(self, ch, num_heads, dim_head, depth, context_dim):
        super().__init__()
        self.image_spatial_transformer = SpatialTransformer(
            ch, num_heads, dim_head, depth, context_dim=context_dim)
        self.st_spatial_transformer = STSpatialTransformer(
            ch, num_heads, dim_head, depth, context_dim=context_dim)

    def forward(self, image, st, context4image=None, context4st=None):
        image = self.image_spatial_transformer(image, context=context4image)
        st = self.st_spatial_transformer(st, context=context4st)

        return image, st

class DualSplitJointTransformerv2(nn.Module):

    def __init__(self, ch, num_heads, dim_head, depth, context_dim):
        super().__init__()
        self.image_spatial_transformer = SpatialTransformer(
            ch, num_heads, dim_head, depth, context_dim=context_dim)
        self.st_spatial_transformer = STSpatialTransformer(
            ch, num_heads, dim_head, depth, context_dim=context_dim)

    def forward(self, image, st, context4image=None, context4st=None):
        if context4image is not None:
            fusion_token = torch.cat([context4image, context4st], dim=1)
        image = self.image_spatial_transformer(image, context=fusion_token)
        st = self.st_spatial_transformer(st, context=fusion_token)

        return image, st


class BasicTransformerBlock3D(nn.Module):

    def __init__(
            self,
            dim,
            n_heads,
            d_head,
            context_dim,
            dropout=0.0,
            gated_ff=True,
            ip_dim=0,
            ip_weight=1,
    ):
        super().__init__()

        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None, num_frames=1):
        x = rearrange(x, "(b f) l c -> b (f l) c", f=num_frames).contiguous()
        x = self.attn1(self.norm1(x), context=None) + x
        x = rearrange(x, "b (f l) c -> (b f) l c", f=num_frames).contiguous()
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class STSpatialTransformer3D(nn.Module):

    def __init__(
            self,
            in_channels,
            n_heads,
            d_head,
            depth=1,
            context_dim=None,
            dropout=0.0,
            ip_dim=0,
            ip_weight=1,
    ):
        super().__init__()

        self.in_channels = in_channels

        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.st_proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.st_transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock3D(
                    inner_dim,
                    n_heads,
                    d_head,
                    context_dim=context_dim,
                    dropout=dropout,
                    ip_dim=ip_dim,
                    ip_weight=ip_weight,
                )
                for d in range(depth)
            ]
        )
        self.st_proj_out = zero_module(nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0 ))

    def forward(self, x, context=None, num_frames=1):
        b, c, t = x.shape
        st_in = x

        st = self.st_proj_in(self.norm(x))
        st = rearrange(x, "b c t -> b t c").contiguous()
        for st_block in self.st_transformer_blocks:
            st = st_block(st, context=context, num_frames=num_frames)
        st = rearrange(st, "b t c -> b c t").contiguous()
        st = self.st_proj_out(st)


        return st + st_in